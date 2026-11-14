import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)
    k = min(5, n - 1)
    nearest = np.sort(work, axis=1)[:, :k]
    finite_nearest = np.isfinite(nearest)

    counts = finite_nearest.sum(axis=1).astype(float)
    safe_nearest = np.where(finite_nearest, nearest, 0.0)

    local_scale = np.divide(
        safe_nearest.sum(axis=1),
        np.maximum(counts, 1.0),
        out=np.ones(n, dtype=float),
        where=counts > 0.0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    local_std = np.zeros(n, dtype=float)
    for i in range(n):
        values = nearest[i, finite_nearest[i]]
        if values.size > 1:
            local_std[i] = np.std(values) / max(local_scale[i], np.finfo(float).eps)

    adaptive_decay = np.clip(0.90 + 0.35 * local_std, 0.75, 1.45)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    distance_quality = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
        / adaptive_decay[:, None]
    )

    row_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                row_scale,
                out=np.full_like(d, np.inf, dtype=float),
                where=valid,
            ),
            0.0,
            50.0,
        )
        / adaptive_decay[:, None]
    )
    col_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                col_scale,
                out=np.full_like(d, np.inf, dtype=float),
                where=valid,
            ),
            0.0,
            50.0,
        )
        / adaptive_decay[None, :]
    )

    mutual_quality = np.sqrt(row_quality * col_quality)

    best = nearest[:, 0]
    second = nearest[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.20 * regret

    row_ranks = np.zeros_like(d, dtype=float)
    col_ranks = np.zeros_like(d, dtype=float)

    row_order = np.argsort(np.where(valid, d, np.inf), axis=1)
    row_rank_values = np.argsort(row_order, axis=1) + 1.0
    row_ranks[valid] = row_rank_values[valid]

    col_order = np.argsort(np.where(valid, d, np.inf), axis=0)
    col_rank_values = np.argsort(col_order, axis=0) + 1.0
    col_ranks[valid] = col_rank_values[valid]

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.06 * row_ranks) * (1.0 + 0.06 * col_ranks)
    )

    row_competition = np.sum(
        valid & (d <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (d <= safe_second[None, :]),
        axis=0,
        dtype=float,
    )

    endpoint_scarcity = 1.0 + 0.08 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    row_degree = valid.sum(axis=1).astype(float)
    col_degree = valid.sum(axis=0).astype(float)

    degree_support = np.sqrt(
        np.maximum(row_degree[:, None], 1.0)
        * np.maximum(col_degree[None, :], 1.0)
    )
    degree_potential = 1.0 + 0.06 / np.sqrt(degree_support)

    score = (
        distance_quality
        * mutual_quality
        * rank_quality
        * endpoint_scarcity
        * degree_potential
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )

    valid_counts = valid.sum(axis=1, keepdims=True).astype(float)
    safe_probabilities = np.where(probabilities > 0.0, probabilities, 1.0)
    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(safe_probabilities), 0.0),
        axis=1,
        keepdims=True,
    )
    max_entropy = np.log(np.maximum(valid_counts, 2.0))

    concentration = 1.0 - np.divide(
        entropy,
        max_entropy,
        out=np.zeros_like(entropy),
        where=max_entropy > 0.0,
    )
    exploration = np.clip(0.035 + 0.075 * concentration, 0.035, 0.11)

    uniform_valid = np.divide(
        valid.astype(float),
        np.maximum(valid_counts, 1.0),
    )

    result = (1.0 - exploration) * probabilities + exploration * uniform_valid
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    final_totals = result.sum(axis=1, keepdims=True)
    return np.divide(
        result,
        final_totals,
        out=np.zeros_like(result),
        where=final_totals > 0.0,
    )
