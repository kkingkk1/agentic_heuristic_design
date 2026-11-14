import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, normalized edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid = finite_nonnegative & off_diagonal
    symmetric_valid = valid & valid.T

    if not np.any(symmetric_valid):
        return result

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance = np.maximum(symmetric_distance, 0.0)

    valid_values = symmetric_distance[symmetric_valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        fallback = float(np.percentile(positive_values, 90.0))
        if not np.isfinite(fallback) or fallback <= 0.0:
            fallback = float(np.max(positive_values))
        scale_floor = max(
            float(np.median(positive_values)) * 1e-10,
            np.finfo(float).eps,
        )
    else:
        fallback = 1.0
        scale_floor = np.finfo(float).eps

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    working_distance = np.where(
        symmetric_valid,
        symmetric_distance,
        fallback,
    ).astype(float, copy=False)
    np.fill_diagonal(working_distance, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(working_distance, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(
        np.nan_to_num(
            local_scale,
            nan=fallback,
            posinf=fallback,
            neginf=fallback,
        ),
        scale_floor,
    )

    sorted_distance = np.sort(working_distance, axis=1)
    nearest_distance = sorted_distance[:, 0]

    if n >= 3:
        alternative_distance = sorted_distance[:, 1]
    else:
        alternative_distance = nearest_distance.copy()

    alternative_distance = np.where(
        np.isfinite(alternative_distance),
        alternative_distance,
        local_scale,
    )
    alternative_distance = np.maximum(alternative_distance, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = working_distance / (pair_scale + scale_floor)

    distance_affinity = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    endpoint_affinity_i = np.exp(
        -np.clip(
            working_distance / (local_scale[:, None] + scale_floor),
            0.0,
            50.0,
        )
    )
    endpoint_affinity_j = np.exp(
        -np.clip(
            working_distance / (local_scale[None, :] + scale_floor),
            0.0,
            50.0,
        )
    )
    endpoint_affinity = np.sqrt(
        endpoint_affinity_i * endpoint_affinity_j
    )

    order = np.argsort(
        working_distance,
        axis=1,
        kind="mergesort",
    )
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))
    rank_affinity_i = np.exp(-ranks / rank_scale)
    rank_affinity_j = rank_affinity_i.T
    mutual_rank_affinity = np.sqrt(
        rank_affinity_i * rank_affinity_j
    )

    alternative_scale = np.sqrt(
        alternative_distance[:, None]
        * alternative_distance[None, :]
    )
    replacement_margin = (
        alternative_scale - working_distance
    ) / (
        alternative_scale + working_distance + scale_floor
    )
    replacement_margin = np.clip(replacement_margin, 0.0, 1.0)

    nearest_bonus = np.sqrt(
        np.clip(
            nearest_distance[:, None]
            / (working_distance + scale_floor),
            0.0,
            1.0,
        )
    )
    nearest_bonus_j = np.sqrt(
        np.clip(
            nearest_distance[None, :]
            / (working_distance + scale_floor),
            0.0,
            1.0,
        )
    )
    endpoint_specificity = np.sqrt(nearest_bonus * nearest_bonus_j)

    rank_gate = 0.55 + 0.45 * np.sqrt(
        np.clip(mutual_rank_affinity, 0.0, 1.0)
    )
    margin_bonus = 1.0 + 0.14 * replacement_margin * rank_gate
    specificity_bonus = 0.90 + 0.10 * endpoint_specificity

    score = (
        distance_affinity
        * endpoint_affinity
        * mutual_rank_affinity
        * margin_bonus
        * specificity_bonus
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
