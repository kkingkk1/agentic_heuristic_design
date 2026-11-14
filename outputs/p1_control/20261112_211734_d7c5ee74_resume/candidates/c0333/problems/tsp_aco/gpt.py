import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_offdiag = d[np.isfinite(d)]
    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_offdiag))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    positive_scale = float(np.median(positive))
    scale_floor = max(positive_scale * 1e-10, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n) + 0.5))))

    local_indices = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    local_distances = np.take_along_axis(d, local_indices, axis=1)
    local_distances = np.maximum(local_distances, scale_floor)

    local_weights = np.linspace(1.0, 0.65, k, dtype=float)
    local_weights /= np.sum(local_weights)

    local_scale = local_distances @ local_weights
    local_scale = np.maximum(local_scale, scale_floor)

    local_mean = np.mean(local_distances, axis=1)
    local_std = np.std(local_distances, axis=1)
    dispersion = local_std / np.maximum(local_mean, scale_floor)
    dispersion_factor = dispersion / (1.0 + dispersion)

    adaptive_width = k * (0.90 + 0.45 * dispersion_factor)
    adaptive_width = np.maximum(adaptive_width, 1.0)

    if n > 2:
        second_partition = np.partition(d, kth=1, axis=1)
        alternative = second_partition[:, 1]
    else:
        alternative = local_distances[:, 0]

    alternative = np.maximum(alternative, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / (pair_scale + scale_floor)

    distance_affinity = np.exp(
        -np.clip(relative_distance ** 0.94, 0.0, 50.0)
    )

    endpoint_i = np.exp(
        -np.clip(
            (d / (local_scale[:, None] + scale_floor)) ** 0.90,
            0.0,
            50.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            (d / (local_scale[None, :] + scale_floor)) ** 0.90,
            0.0,
            50.0,
        )
    )

    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)
    bottleneck_compatibility = np.minimum(endpoint_i, endpoint_j)

    local_order = np.argsort(local_distances, axis=1, kind="stable")
    local_ranks = np.empty_like(local_order, dtype=float)
    local_ranks[
        np.arange(n)[:, None],
        local_order,
    ] = np.arange(k, dtype=float)[None, :]

    rank_score = np.full(
        (n, n),
        1.0 / np.sqrt(1.0 + float(k)),
        dtype=float,
    )
    rows = np.arange(n)[:, None]
    rank_score[rows, local_indices] = 1.0 / np.sqrt(1.0 + local_ranks)

    mutual_rank = np.sqrt(rank_score * rank_score.T)

    rank_support_i = np.exp(
        -np.clip(
            np.minimum(local_ranks, float(k)) / adaptive_width[:, None],
            0.0,
            50.0,
        )
    )

    full_rank_support_i = np.full(
        (n, n),
        np.exp(-float(k) / np.mean(adaptive_width)),
        dtype=float,
    )
    full_rank_support_i[rows, local_indices] = rank_support_i

    candidate_support = np.sqrt(
        full_rank_support_i * full_rank_support_i.T
    )
    candidate_gate = 0.78 + 0.22 * candidate_support

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )

    margin = alternative_scale / (
        d + alternative_scale + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_bonus = 1.0 + 0.08 * margin

    density_ratio = np.sqrt(
        local_scale[:, None]
        / np.maximum(local_scale[None, :], scale_floor)
    )
    density_balance = np.exp(
        -0.14 * np.abs(np.log(np.maximum(density_ratio, eps)))
    )

    global_scale = max(float(np.median(positive)), scale_floor)
    global_guard = 0.93 + 0.07 * np.exp(
        -np.clip((d / global_scale) ** 0.92, 0.0, 50.0)
    )

    score = (
        distance_affinity
        * endpoint_compatibility
        * bottleneck_compatibility
        * mutual_rank
        * candidate_gate
        * margin_bonus
        * density_balance
        * global_guard
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
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
