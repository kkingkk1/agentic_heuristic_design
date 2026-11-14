import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, softly sparse edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & off_diagonal
    positive_values = distances[finite & (distances > 0.0)]

    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = np.asarray(positive_values, dtype=float)
    median = float(np.median(positive_values))
    scale = max(median, np.finfo(float).eps)

    # Robust bounds reduce the influence of outliers while preserving local order.
    lower_quantile = float(np.percentile(positive_values, 5.0))
    upper_quantile = float(np.percentile(positive_values, 95.0))
    lower = max(lower_quantile, scale * 1e-6, np.finfo(float).eps)
    upper = max(upper_quantile, lower)

    # Non-positive finite values are treated as extremely short, but bounded, edges.
    safe = np.where(
        finite,
        np.where(distances > 0.0, distances, lower),
        upper,
    )
    safe = np.clip(safe, lower, upper)
    safe[~off_diagonal] = np.inf

    usable = finite
    if not np.any(usable):
        return np.zeros((n, n), dtype=float)

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
        valid_row = usable[i, row_order[i]]
        row_rank[i, row_order[i][valid_row]] = rank_values[valid_row]

        valid_col = usable[col_order[:, i], i]
        col_rank[col_order[:, i][valid_col], i] = rank_values[valid_col]

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

    # Reciprocal nearest-neighbor bonus, excluding self-loops by construction.
    reciprocal = (
        (row_rank <= 0.0) & (col_rank <= 0.0) & usable
    ).astype(float)

    # Keep a small adaptive neighborhood while retaining a soft exploration floor.
    candidate_count = min(
        max(3, int(np.ceil(np.sqrt(n)))),
        n - 1,
    )
    candidate_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = row_order[i, :candidate_count]
        candidates = candidates[usable[i, candidates]]
        candidate_mask[i, candidates] = True

    priors = (
        0.52 * inverse
        + 0.16 * row_rank_score
        + 0.14 * col_rank_score
        + 0.10 * row_competitiveness
        + 0.08 * col_competitiveness
    )
    priors *= 1.0 + 0.20 * reciprocal

    # Softly suppress distant candidates instead of deleting them.
    priors *= np.where(candidate_mask, 1.0, 0.08)
    priors[~usable] = 0.0
    np.fill_diagonal(priors, 0.0)

    maximum = float(np.max(priors))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    priors /= maximum
    return np.nan_to_num(priors, nan=0.0, posinf=0.0, neginf=0.0)
