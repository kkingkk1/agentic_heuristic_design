import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return locally normalized edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    positive = distances[
        finite_nonnegative & (distances > 0.0)
    ]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite_nonnegative, distances, fallback)
    d = 0.5 * (d + d.T)
    d = np.maximum(d, 0.0)

    usable = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(usable, False)

    positive_scale = float(np.median(positive))
    eps = np.finfo(float).eps
    floor = max(positive_scale * 1e-12, eps)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    ordered = np.sort(work, axis=1)

    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, floor)

    nearest = ordered[:, 0]
    second_nearest = (
        ordered[:, 1] if n > 2 else ordered[:, 0]
    )
    nearest = np.maximum(nearest, floor)
    second_nearest = np.maximum(second_nearest, floor)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / (endpoint_scale + floor)
    distance_affinity = 1.0 / (
        1.0 + np.minimum(relative_distance, 1e12)
    )

    endpoint_i_affinity = 1.0 / (
        1.0 + d / (local_scale[:, None] + floor)
    )
    endpoint_j_affinity = 1.0 / (
        1.0 + d / (local_scale[None, :] + floor)
    )
    endpoint_affinity = np.sqrt(
        endpoint_i_affinity * endpoint_j_affinity
    )

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]

    rank_affinity = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_affinity * rank_affinity.T)

    regret_i = np.clip(
        (second_nearest[:, None] - d)
        / (second_nearest[:, None] + d + floor),
        -1.0,
        1.0,
    )
    regret_j = np.clip(
        (second_nearest[None, :] - d)
        / (second_nearest[None, :] + d + floor),
        -1.0,
        1.0,
    )
    regret_bonus = 1.0 + 0.18 * np.maximum(
        0.0, 0.5 * (regret_i + regret_j)
    )

    score = (
        distance_affinity
        * endpoint_affinity
        * np.sqrt(mutual_rank)
        * regret_bonus
    )

    score = 0.5 * (score + score.T)
    score[~usable] = 0.0
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
