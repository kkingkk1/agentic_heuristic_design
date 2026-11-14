import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust soft edge priors for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    usable = np.isfinite(distances) & (distances >= 0.0) & off_diagonal

    if not np.any(usable):
        return result

    positive = distances[usable & (distances > 0.0)]

    if positive.size:
        positive = positive[np.isfinite(positive)]

        q05 = float(np.percentile(positive, 5.0))
        q50 = float(np.percentile(positive, 50.0))
        q95 = float(np.percentile(positive, 95.0))

        eps = np.finfo(float).eps
        lower = max(q05, q50 * 1e-6, eps)
        upper = max(q95, lower)
        scale = max(q50, lower)
    else:
        lower = 1.0
        upper = 1.0
        scale = 1.0

    safe_distance = np.full((n, n), upper, dtype=float)
    valid_values = distances[usable]

    safe_distance[usable] = np.where(
        valid_values > 0.0,
        np.clip(valid_values, lower, upper),
        lower,
    )

    normalized_distance = safe_distance / scale

    inverse = np.zeros((n, n), dtype=float)
    inverse[usable] = 1.0 / (normalized_distance[usable] + 1e-12)

    order_values = np.where(usable, safe_distance, np.inf)
    row_order = np.argsort(order_values, axis=1, kind="stable")
    col_order = np.argsort(order_values, axis=0, kind="stable")

    row_rank = np.full((n, n), float(n), dtype=float)
    col_rank = np.full((n, n), float(n), dtype=float)

    rank_values = np.arange(n, dtype=float)

    for i in range(n):
        row_indices = row_order[i]
        valid_row = usable[i, row_indices]
        row_rank[i, row_indices[valid_row]] = rank_values[valid_row]

        col_indices = col_order[:, i]
        valid_col = usable[col_indices, i]
        col_rank[col_indices[valid_col], i] = rank_values[valid_col]

    denominator = float(max(n - 1, 1))

    row_rank_score = np.exp(-row_rank / denominator)
    col_rank_score = np.exp(-col_rank / denominator)

    row_best = np.max(np.where(usable, inverse, 0.0), axis=1, keepdims=True)
    col_best = np.max(np.where(usable, inverse, 0.0), axis=0, keepdims=True)

    row_competitiveness = np.divide(
        inverse,
        row_best,
        out=np.zeros_like(inverse),
        where=row_best > 0.0,
    )
    col_competitiveness = np.divide(
        inverse,
        col_best,
        out=np.zeros_like(inverse),
        where=col_best > 0.0,
    )

    positive_inverse = inverse[usable]
    global_reference = float(np.percentile(positive_inverse, 90.0))
    global_reference = max(global_reference, np.finfo(float).eps)

    global_quality = np.clip(inverse / global_reference, 0.0, 1.0)

    candidate_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n)))),
    )

    candidate_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = row_order[i, :candidate_count]
        candidates = candidates[usable[i, candidates]]
        candidate_mask[i, candidates] = True

    reciprocal = (
        usable
        & (row_rank <= 1.0)
        & (col_rank <= 1.0)
    )

    priors = (
        0.54 * inverse
        + 0.14 * row_rank_score
        + 0.12 * col_rank_score
        + 0.09 * row_competitiveness
        + 0.07 * col_competitiveness
        + 0.04 * global_quality
    )

    priors *= 1.0 + 0.16 * reciprocal
    priors *= np.where(candidate_mask, 1.0, 0.10)

    priors[~usable] = 0.0
    np.fill_diagonal(priors, 0.0)

    maximum = float(np.max(priors))

    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    priors /= maximum

    return np.nan_to_num(
        priors,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
