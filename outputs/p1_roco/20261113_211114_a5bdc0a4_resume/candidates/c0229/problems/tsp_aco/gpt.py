import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    finite_mask = np.isfinite(symmetric_distance)
    finite_values = symmetric_distance[finite_mask]

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    safe_distance = np.where(finite_mask, symmetric_distance, 0.0)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe_distance[valid] + eps)

    inverse_max = np.max(inverse, axis=1, keepdims=True)
    relative_inverse = inverse / np.maximum(inverse_max, eps)

    rank_score = np.zeros((n, n), dtype=float)
    valid_counts = valid.sum(axis=1)

    for i in range(n):
        count = int(valid_counts[i])
        if count == 0:
            continue

        row = symmetric_distance[i]
        order = np.argsort(row, kind="stable")
        neighbors = order[valid[i, order]]
        ranks = np.arange(count, dtype=float)

        rank_score[i, neighbors] = 1.0 - ranks / max(float(count - 1), 1.0)

    mutual_rank = np.sqrt(np.maximum(rank_score * rank_score.T, 0.0))

    temperature = max(0.75 * scale, eps)
    soft_support = np.zeros((n, n), dtype=float)
    soft_support[valid] = np.exp(
        -np.minimum(safe_distance[valid] / temperature, 700.0)
    )

    support_mass = soft_support.sum(axis=1)
    normalized_support = soft_support / np.maximum(support_mass[:, None], eps)
    mutual_support = np.sqrt(
        np.maximum(normalized_support * normalized_support.T, 0.0)
    )

    scarcity = 1.0 / np.sqrt(np.maximum(support_mass, eps))
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.5, 2.0)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )

    score = inverse
    score *= 0.45 + 0.55 * relative_inverse
    score *= 0.55 + 0.75 * mutual_rank
    score *= 0.80 + 0.35 * mutual_support
    score *= 0.88 + 0.22 * endpoint_scarcity

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
