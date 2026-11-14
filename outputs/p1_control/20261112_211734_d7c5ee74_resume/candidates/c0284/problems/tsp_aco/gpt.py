import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, finite, normalized edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[valid]
    fallback = float(np.median(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    offdiag = d[~np.eye(n, dtype=bool)]
    finite_offdiag = offdiag[np.isfinite(offdiag) & (offdiag > 0.0)]
    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(finite_offdiag))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    positive = d[np.isfinite(d) & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else fallback
    global_scale = max(global_scale, eps)

    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))

    ordered = np.sort(d, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else nearest.copy()
    local = ordered[:, :k]

    weights = np.linspace(1.0, 0.55, k, dtype=float)
    weights /= weights.sum()
    local_scale = np.maximum(local @ weights, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = d / np.maximum(endpoint_scale, eps)
    distance_score = 1.0 / (
        1.0 + np.power(np.maximum(normalized_distance, 0.0), 1.15)
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(float(k - 1), 1.0)
    rank_i = np.exp(-np.minimum(row_rank / rank_scale, 8.0))
    rank_j = rank_i.T
    rank_score = np.sqrt(rank_i * rank_j)

    candidate_support = 0.5 * (
        (row_rank <= (k - 1)).astype(float)
        + (row_rank.T <= (k - 1)).astype(float)
    )
    candidate_gate = 0.68 + 0.32 * candidate_support

    gap = np.maximum(second - nearest, eps)

    margin_i = np.clip(
        (second[:, None] - d) / np.maximum(second[:, None] + gap[:, None], eps),
        0.0,
        1.0,
    )
    margin_j = np.clip(
        (second[None, :] - d) / np.maximum(second[None, :] + gap[None, :], eps),
        0.0,
        1.0,
    )
    margin_score = 1.0 + 0.16 * np.sqrt(margin_i * margin_j)

    density_ratio = local_scale[:, None] / np.maximum(local_scale[None, :], eps)
    density_score = np.exp(
        -0.08 * np.abs(np.log(np.maximum(density_ratio, eps)))
    )

    global_score = 0.90 + 0.10 * np.exp(
        -np.minimum(np.power(d / global_scale, 0.8), 30.0)
    )

    score = (
        distance_score
        * (0.55 + 0.45 * rank_score)
        * candidate_gate
        * margin_score
        * density_score
        * global_score
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
