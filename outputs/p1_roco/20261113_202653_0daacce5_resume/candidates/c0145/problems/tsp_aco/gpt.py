import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0

    valid_one_way = finite & nonnegative & ~diagonal
    valid = valid_one_way & valid_one_way.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(valid_one_way, np.maximum(raw, 0.0), 0.0)
    distances = 0.5 * (cleaned + cleaned.T)
    distances[~valid] = np.inf
    np.fill_diagonal(distances, np.inf)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    finite_mask = np.isfinite(distances)
    nearest = np.min(
        np.where(finite_mask, distances, np.inf),
        axis=1,
    )
    nearest[~np.isfinite(nearest)] = scale

    baseline = nearest[:, None] + nearest[None, :]
    savings = np.maximum(baseline - distances, 0.0)
    savings[~valid] = 0.0

    max_savings = float(np.max(savings))
    if max_savings > eps and np.isfinite(max_savings):
        savings_score = savings / max_savings
    else:
        savings_score = np.zeros((n, n), dtype=float)

    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_valid = valid[i]
        count = int(np.sum(row_valid))
        if count == 0:
            continue

        values = distances[i].copy()
        values[~row_valid] = np.inf
        order = np.argsort(values, kind="stable")

        position = 0
        for j in order:
            if row_valid[j]:
                ranks[i, j] = 1.0 - (
                    position / max(count - 1, 1)
                )
                position += 1

    rank_support = np.sqrt(
        np.maximum(ranks * ranks.T, 0.0)
    )
    rank_support[~valid] = 0.0

    proximity = np.zeros((n, n), dtype=float)
    proximity[valid] = np.exp(
        -distances[valid] / max(scale, eps)
    )

    triangle_support = np.zeros((n, n), dtype=float)
    for k in range(n):
        left = proximity[:, k, None]
        right = proximity[None, k, :]
        triangle_support = np.maximum(
            triangle_support,
            np.sqrt(np.maximum(left * right, 0.0)),
        )

    triangle_support[~valid] = 0.0
    np.fill_diagonal(triangle_support, 0.0)

    endpoint_quality = np.sqrt(
        np.maximum(
            np.maximum(savings_score, 0.0)
            * np.maximum(rank_support, 0.0),
            0.0,
        )
    )

    local_degree = np.sum(valid, axis=1).astype(float)
    coverage = np.zeros((n, n), dtype=float)
    coverage[valid] = 1.0 / np.sqrt(
        np.maximum(local_degree[:, None] * local_degree[None, :], 1.0)
    )
    coverage[valid] /= max(float(np.max(coverage)), 1.0)

    score = (
        0.46 * endpoint_quality
        + 0.30 * triangle_support
        + 0.16 * rank_support
        + 0.08 * coverage
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
