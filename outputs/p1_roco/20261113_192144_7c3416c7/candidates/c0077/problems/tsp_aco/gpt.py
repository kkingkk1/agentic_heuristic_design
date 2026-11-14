import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, softly sparse edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    usable = np.isfinite(distances) & (distances >= 0.0) & off_diagonal

    positive = distances[usable & (distances > 0.0)]
    if positive.size:
        scale = max(float(np.median(positive)), np.finfo(float).eps)
        lower = max(
            float(np.percentile(positive, 5.0)),
            scale * 1e-6,
            np.finfo(float).eps,
        )
        upper = max(
            float(np.percentile(positive, 95.0)),
            lower,
        )
    elif np.any(usable):
        scale = 1.0
        lower = np.finfo(float).eps
        upper = 1.0
    else:
        return np.zeros((n, n), dtype=float)

    safe = np.full((n, n), upper, dtype=float)
    safe[usable] = np.where(
        distances[usable] > 0.0,
        distances[usable],
        lower,
    )
    safe = np.clip(safe, lower, upper)

    normalized = safe / scale
    inverse = np.zeros((n, n), dtype=float)
    inverse[usable] = 1.0 / (normalized[usable] + 1e-12)

    order_values = np.where(usable, safe, np.inf)
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

    global_best = float(np.max(np.where(usable, inverse, 0.0)))
    global_quality = (
        inverse / global_best
        if global_best > 0.0
        else np.zeros_like(inverse)
    )

    reciprocal = (
        usable
        & (row_rank == 0.0)
        & (col_rank == 0.0)
    )

    candidate_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n)))),
    )
    candidate_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = row_order[i, :candidate_count]
        candidates = candidates[usable[i, candidates]]
        candidate_mask[i, candidates] = True

    priors = (
        0.50 * inverse
        + 0.15 * row_rank_score
        + 0.13 * col_rank_score
        + 0.09 * row_competitiveness
        + 0.08 * col_competitiveness
        + 0.05 * global_quality
    )

    priors *= 1.0 + 0.20 * reciprocal
    priors *= np.where(candidate_mask, 1.0, 0.08)
    priors[~usable] = 0.0
    np.fill_diagonal(priors, 0.0)

    maximum = float(np.max(priors))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    priors /= maximum
    return np.nan_to_num(
        priors,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
