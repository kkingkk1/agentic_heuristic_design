import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return result

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T

    working = np.where(valid, distances, fallback)
    working = 0.5 * (working + working.T)
    working = np.maximum(working, 0.0)
    np.fill_diagonal(working, np.inf)

    positive_scale = float(np.median(positive))
    eps = np.finfo(float).eps
    scale_floor = max(positive_scale * 1e-10, eps)

    neighbor_count = max(
        1,
        min(n - 1, int(np.ceil(np.sqrt(n) + 0.5))),
    )

    nearest = np.partition(
        working,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        scale_floor,
    )

    if n > 2:
        second_nearest = np.partition(
            working,
            kth=1,
            axis=1,
        )[:, 1]
    else:
        second_nearest = nearest[:, 0]

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_pair = working / (pair_scale + eps)
    normalized_left = working / (local_scale[:, None] + eps)
    normalized_right = working / (local_scale[None, :] + eps)

    exponent = (
        normalized_pair
        + 0.5 * normalized_left
        + 0.5 * normalized_right
    )

    affinity = np.exp(-np.clip(exponent, 0.0, 120.0))

    # Fast rank-aware approximation avoiding full row-wise sorting.
    rank_left = 1.0 / np.sqrt(
        1.0 + working / (local_scale[:, None] + eps)
    )
    rank_right = 1.0 / np.sqrt(
        1.0 + working / (local_scale[None, :] + eps)
    )
    mutual_rank = np.sqrt(rank_left * rank_right)

    alternative_scale = np.sqrt(
        second_nearest[:, None] * second_nearest[None, :]
    )

    margin = alternative_scale / (
        working + alternative_scale + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    score = affinity * mutual_rank
    score *= 1.0 + 0.08 * margin
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
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
