import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized edge-priority matrix for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite_positive = np.isfinite(distances) & (distances > 0)
    valid = finite_positive & finite_positive.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    positive_values = d[np.isfinite(d)]
    global_scale = float(np.median(positive_values))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    ordered = np.sort(d, axis=1)
    finite_rows = np.isfinite(ordered).any(axis=1)

    k = min(7, n - 1)
    nearest = ordered[:, :k]
    finite_nearest = np.isfinite(nearest)

    local_scale = np.divide(
        np.where(finite_nearest, nearest, 0.0).sum(axis=1),
        finite_nearest.sum(axis=1),
        out=np.full(n, global_scale, dtype=float),
        where=finite_nearest.sum(axis=1) > 0,
    )
    local_scale = np.where(
        finite_rows & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    if n > 2:
        best = ordered[:, 0]
        second = ordered[:, 1]
    else:
        best = ordered[:, 0]
        second = best.copy()

    best = np.where(np.isfinite(best), best, global_scale)
    second = np.where(np.isfinite(second), second, 2.0 * best)

    regret = np.divide(
        second - best,
        np.maximum(second, np.finfo(float).eps),
    )
    regret = np.clip(np.nan_to_num(regret), 0.0, 1.0)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    clipped_relative = np.clip(relative_distance, 0.0, 60.0)

    distance_term = np.exp(-clipped_relative)
    long_edge_penalty = np.exp(
        -0.35 * np.maximum(relative_distance - 1.0, 0.0)
    )

    row_logits = np.divide(
        -d,
        local_scale[:, None],
        out=np.full((n, n), -np.inf, dtype=float),
        where=valid,
    )
    row_quality = np.exp(np.clip(row_logits, -60.0, 0.0))
    row_quality[~valid] = 0.0

    row_totals = row_quality.sum(axis=1, keepdims=True)
    row_probability = np.divide(
        row_quality,
        row_totals,
        out=np.zeros_like(row_quality),
        where=row_totals > 0.0,
    )

    mutual_choice = np.sqrt(row_probability * row_probability.T)

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)

    col_order = np.argsort(d, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_term = np.exp(
        -0.5 * (row_rank + col_rank) / rank_scale
    )
    rank_term[~valid] = 0.0

    accessibility = np.exp(
        -np.divide(
            d,
            local_scale[:, None],
            out=np.full((n, n), np.inf, dtype=float),
            where=valid,
        )
    )
    accessibility[~valid] = 0.0

    endpoint_access = accessibility.sum(axis=1)
    scarcity = 1.0 / np.sqrt(
        1.0 + endpoint_access[:, None] + endpoint_access[None, :]
    )

    regret_term = 1.0 + 0.40 * (
        regret[:, None] + regret[None, :]
    )

    score = (
        distance_term
        * (0.55 + 0.45 * mutual_choice)
        * (0.65 + 0.35 * rank_term)
        * regret_term
        * (1.0 + 0.35 * scarcity)
        * long_edge_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
