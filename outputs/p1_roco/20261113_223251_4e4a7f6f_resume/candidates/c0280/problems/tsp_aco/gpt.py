import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structure-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    finite_max = float(np.max(observed))
    penalty = finite_max + 8.0 * scale

    work = np.where(valid, distances, penalty)
    np.fill_diagonal(work, penalty)

    temperature = max(0.42 * scale, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.clip(work[valid] / temperature, 0.0, 60.0)
    )

    row_mass = affinity.sum(axis=1)
    local_probability = affinity / np.maximum(row_mass[:, None], eps)

    mutual_preference = np.sqrt(
        np.maximum(local_probability * local_probability.T, 0.0)
    )

    valid_degree = valid.sum(axis=1).astype(float)
    degree_factor = np.sqrt(
        np.maximum(valid_degree[:, None] * valid_degree[None, :], 0.0)
    )
    degree_factor /= max(float(np.max(degree_factor)), 1.0)

    scarcity = 1.0 / np.sqrt(np.maximum(row_mass, eps))
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.50, 2.00)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )

    bypass = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        candidate = work[:, k, None] + work[None, k, :]
        bypass = np.minimum(bypass, candidate)

    np.fill_diagonal(bypass, penalty)

    bypass_advantage = np.maximum(bypass - work, 0.0)
    irreplaceability = np.zeros((n, n), dtype=float)
    irreplaceability[valid] = 1.0 - np.exp(
        -bypass_advantage[valid] / max(0.72 * scale, eps)
    )

    triangle_support = np.zeros((n, n), dtype=float)
    for k in range(n):
        two_hop = work[:, k, None] + work[None, k, :]
        closure = np.exp(
            -np.clip(
                np.maximum(two_hop - work, 0.0) / max(0.85 * scale, eps),
                0.0,
                60.0,
            )
        )
        triangle_support = np.maximum(triangle_support, closure)

    triangle_support[~valid] = 0.0

    base = affinity / max(float(np.max(affinity)), eps)
    mutual = mutual_preference / max(
        float(np.max(mutual_preference)),
        eps,
    )

    score = base
    score *= 0.58 + 0.42 * mutual
    score *= 0.74 + 0.30 * endpoint_scarcity
    score *= 0.86 + 0.18 * degree_factor
    score *= 0.76 + 0.24 * irreplaceability
    score *= 0.82 + 0.18 * triangle_support

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()
    for _ in range(40):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.68 * score + 0.32 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
