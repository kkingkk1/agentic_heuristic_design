import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    work = np.where(positive, d, np.inf)
    np.fill_diagonal(work, np.inf)

    valid_values = work[np.isfinite(work)]
    if valid_values.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = np.median(valid_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    eps = np.finfo(float).eps

    # Multi-scale short-edge preference.
    scales = np.array([0.35, 0.8, 1.8]) * scale
    kernels = np.exp(-work[..., None] / (scales[None, None, :] + eps))
    shortness = np.mean(kernels, axis=2)

    # Endpoint agreement: an edge is favored when both endpoints rank it highly.
    row_rank = np.full((n, n), n, dtype=float)
    col_rank = np.full((n, n), n, dtype=float)

    for i in range(n):
        order = np.argsort(work[i], kind="stable")
        row_rank[i, order] = np.arange(n, dtype=float)

    for j in range(n):
        order = np.argsort(work[:, j], kind="stable")
        col_rank[order, j] = np.arange(n, dtype=float)

    rank_scale = max(1.0, np.sqrt(n))
    mutual_preference = np.exp(-(row_rank + col_rank) / rank_scale)
    mutual_preference[~positive] = 0.0

    # Neighborhood compatibility through shared inexpensive intermediate nodes.
    normalized = np.where(np.isfinite(work), work / (scale + eps), np.inf)
    affinity = np.exp(-normalized)
    np.fill_diagonal(affinity, 0.0)
    neighborhood_support = affinity @ affinity
    neighborhood_support /= np.maximum(
        np.max(neighborhood_support), eps
    )
    neighborhood_support = np.maximum(neighborhood_support, neighborhood_support.T)
    neighborhood_support[~positive] = 0.0

    # Triangle support: prefer edges participating in short two-hop alternatives.
    two_hop = np.min(work[:, :, None] + work[None, :, :], axis=1)
    triangle_support = np.exp(-np.minimum(work, two_hop) / (scale + eps))
    triangle_support[~positive] = 0.0

    # Insertion-regret signal: large local distance gaps make an edge more useful.
    sorted_rows = np.sort(work, axis=1)
    nearest = sorted_rows[:, 0]
    second_nearest = sorted_rows[:, 1] if n > 2 else nearest
    regret = np.maximum(second_nearest - nearest, 0.0)
    regret_scale = np.median(regret[np.isfinite(regret)])
    if not np.isfinite(regret_scale) or regret_scale <= 0:
        regret_scale = scale
    endpoint_regret = 1.0 - np.exp(
        -(regret[:, None] + regret[None, :]) / (regret_scale + eps)
    )
    endpoint_regret[~positive] = 0.0

    score = (
        0.34 * shortness
        + 0.26 * mutual_preference
        + 0.20 * neighborhood_support
        + 0.12 * triangle_support
        + 0.08 * endpoint_regret
    )

    score = 0.5 * (score + score.T)
    score[~positive] = 0.0

    maximum = np.max(score)
    if maximum > 0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
