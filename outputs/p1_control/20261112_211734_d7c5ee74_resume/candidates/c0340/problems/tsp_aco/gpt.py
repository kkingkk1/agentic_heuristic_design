import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, locally normalized edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    symmetric_valid = finite_nonnegative & finite_nonnegative.T
    positive = distances[finite_nonnegative & (distances > 0.0)]

    if positive.size:
        fallback = float(np.median(positive))
    else:
        fallback = 1.0

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    repaired = np.where(finite_nonnegative, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)

    np.fill_diagonal(repaired, np.inf)

    eps = np.finfo(float).eps
    scale_floor = max(
        float(np.median(positive)) * 1e-10 if positive.size else eps,
        eps,
    )

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n) + 0.5))))

    top_indices = np.argpartition(
        repaired,
        kth=k - 1,
        axis=1,
    )[:, :k]
    top_distances = np.take_along_axis(
        repaired,
        top_indices,
        axis=1,
    )

    local_scale = np.mean(top_distances, axis=1)
    local_scale = np.maximum(local_scale, scale_floor)

    if n > 2:
        second_nearest = np.partition(
            repaired,
            kth=1,
            axis=1,
        )[:, 1]
    else:
        second_nearest = top_distances[:, 0]

    second_nearest = np.maximum(second_nearest, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, scale_floor)

    relative_distance = repaired / pair_scale
    endpoint_relative_i = repaired / np.maximum(
        local_scale[:, None],
        scale_floor,
    )
    endpoint_relative_j = repaired / np.maximum(
        local_scale[None, :],
        scale_floor,
    )

    distance_affinity = np.exp(
        -np.clip(relative_distance ** 0.96, 0.0, 50.0)
    )

    endpoint_affinity_i = np.exp(
        -np.clip(endpoint_relative_i ** 0.90, 0.0, 50.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(endpoint_relative_j ** 0.90, 0.0, 50.0)
    )

    endpoint_compatibility = np.sqrt(
        endpoint_affinity_i * endpoint_affinity_j
    )

    row_order = np.argsort(
        repaired,
        axis=1,
        kind="stable",
    )
    row_rank = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    row_rank[rows, row_order] = np.arange(n, dtype=float)[None, :]

    rank_scale = max(0.85 * k, 1.0)
    rank_affinity = np.exp(
        -0.85
        * np.clip(
            0.5 * (row_rank + row_rank.T) / rank_scale,
            0.0,
            50.0,
        )
    )

    local_mean = np.mean(top_distances, axis=1)
    local_std = np.std(top_distances, axis=1)
    dispersion = local_std / np.maximum(local_mean, scale_floor)

    adaptive_width = k * (
        0.90 + 0.40 * dispersion / (1.0 + dispersion)
    )
    adaptive_width = np.maximum(adaptive_width, 1.0)

    rank_support_i = np.exp(
        -np.clip(
            row_rank / adaptive_width[:, None],
            0.0,
            50.0,
        )
    )
    rank_support_j = np.exp(
        -np.clip(
            row_rank.T / adaptive_width[None, :],
            0.0,
            50.0,
        )
    )

    candidate_support = np.sqrt(
        rank_support_i * rank_support_j
    )
    candidate_factor = 0.80 + 0.20 * candidate_support

    alternative_scale = np.sqrt(
        second_nearest[:, None] * second_nearest[None, :]
    )
    margin = alternative_scale / (
        repaired + alternative_scale + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_factor = 1.0 + 0.08 * margin

    affinity = endpoint_compatibility * distance_affinity
    np.fill_diagonal(affinity, 0.0)

    support = affinity @ affinity.T
    row_norm = np.sqrt(
        np.sum(affinity * affinity, axis=1)
    )
    compatibility = support / (
        row_norm[:, None] * row_norm[None, :] + eps
    )
    compatibility = np.clip(compatibility, 0.0, 1.0)
    compatibility_factor = 0.97 + 0.03 * compatibility

    density_ratio = np.sqrt(
        local_scale[:, None]
        / np.maximum(local_scale[None, :], scale_floor)
    )
    density_factor = np.exp(
        -0.10 * np.abs(
            np.log(np.maximum(density_ratio, eps))
        )
    )

    score = (
        distance_affinity
        * endpoint_compatibility
        * rank_affinity
        * candidate_factor
        * margin_factor
        * compatibility_factor
        * density_factor
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
