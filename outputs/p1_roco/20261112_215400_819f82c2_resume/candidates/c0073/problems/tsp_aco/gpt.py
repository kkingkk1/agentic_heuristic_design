import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_edges = np.isfinite(d)
    values = d[finite_edges]

    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_scale = 1.4826 * mad
    if robust_scale <= np.finfo(float).eps:
        robust_scale = max(median, np.finfo(float).eps)

    positive = values[values > 0.0]
    positive_scale = (
        float(np.median(positive))
        if positive.size
        else np.finfo(float).eps
    )
    global_scale = max(0.5 * robust_scale + 0.5 * positive_scale,
                       np.finfo(float).eps)

    safe_d = np.where(finite_edges, d, 0.0)

    row_values = np.where(finite_edges, d, np.nan)
    local_median = np.nanmedian(row_values, axis=1)
    local_median = np.where(
        np.isfinite(local_median),
        local_median,
        global_scale,
    )
    local_positive = np.where(
        (finite_edges) & (d > 0.0),
        d,
        np.nan,
    )
    local_positive_scale = np.nanmedian(local_positive, axis=1)
    local_positive_scale = np.where(
        np.isfinite(local_positive_scale) &
        (local_positive_scale > np.finfo(float).eps),
        local_positive_scale,
        global_scale,
    )
    local_scale = np.maximum(
        0.5 * local_median + 0.5 * local_positive_scale,
        np.finfo(float).eps,
    )

    local_quality = np.exp(
        -safe_d / local_scale[:, None]
    )
    global_quality = np.exp(-safe_d / global_scale)

    row_rank = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(safe_d[i], kind="mergesort")
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n, dtype=float)
        row_rank[i] = rank

    valid_counts = np.maximum(finite_edges.sum(axis=1) - 1, 1)
    local_rank_quality = np.exp(
        -row_rank / valid_counts[:, None]
    )

    symmetric_quality = 0.5 * (
        local_quality + local_quality.T
    )
    symmetric_rank = 0.5 * (
        local_rank_quality + local_rank_quality.T
    )

    global_order = np.argsort(values, kind="mergesort")
    global_rank_values = np.empty(values.size, dtype=float)
    global_rank_values[global_order] = np.arange(values.size)
    global_rank = np.zeros((n, n), dtype=float)
    global_rank[finite_edges] = global_rank_values
    global_rank_quality = np.exp(
        -global_rank / max(float(values.size), 1.0)
    )

    affinity = np.exp(-safe_d / global_scale)
    np.fill_diagonal(affinity, 0.0)

    row_mass = affinity.sum(axis=1)
    row_mass = np.maximum(row_mass, np.finfo(float).eps)
    normalized_affinity = affinity / row_mass[:, None]

    common_neighbor = normalized_affinity @ normalized_affinity.T
    common_neighbor = np.maximum(common_neighbor, 0.0)
    cn_max = np.max(common_neighbor)
    if cn_max > 0.0:
        common_neighbor /= cn_max

    endpoint_opportunity = np.sqrt(
        np.maximum(row_mass[:, None], 0.0) *
        np.maximum(row_mass[None, :], 0.0)
    )
    opportunity_max = np.max(endpoint_opportunity)
    if opportunity_max > 0.0:
        endpoint_opportunity /= opportunity_max

    score = (
        0.28 * symmetric_quality
        + 0.20 * symmetric_rank
        + 0.17 * global_quality
        + 0.15 * global_rank_quality
        + 0.15 * common_neighbor
        + 0.05 * endpoint_opportunity
    )

    score = 0.5 * (score + score.T)
    score[~finite_edges] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return np.asarray(score, dtype=float)
