import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

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

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)

    work = repaired.copy()
    np.fill_diagonal(work, np.inf)

    scale_floor = max(
        float(np.median(positive)) * 1e-10,
        eps,
    )

    k = max(
        1,
        min(n - 1, int(np.ceil(np.sqrt(n)))),
    )

    top_indices = np.argpartition(
        work,
        kth=k - 1,
        axis=1,
    )[:, :k]

    top_distances = np.take_along_axis(
        work,
        top_indices,
        axis=1,
    )

    local_scale = np.maximum(
        np.mean(top_distances, axis=1),
        scale_floor,
    )

    if n > 2:
        second_neighbors = np.partition(
            work,
            kth=1,
            axis=1,
        )[:, 1]
    else:
        second_neighbors = top_distances[:, 0]

    alternatives = np.where(
        np.isfinite(second_neighbors),
        second_neighbors,
        local_scale,
    )
    alternatives = np.maximum(alternatives, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = work / (pair_scale + scale_floor)
    distance_affinity = np.exp(
        -np.clip(1.05 * normalized_distance, 0.0, 50.0)
    )

    endpoint_i = np.exp(
        -np.clip(
            work / (local_scale[:, None] + scale_floor),
            0.0,
            50.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            work / (local_scale[None, :] + scale_floor),
            0.0,
            50.0,
        )
    )
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    local_order = np.argsort(
        top_distances,
        axis=1,
        kind="mergesort",
    )

    local_ranks = np.empty_like(
        local_order,
        dtype=float,
    )
    local_ranks[
        np.arange(n)[:, None],
        local_order,
    ] = np.arange(k, dtype=float)[None, :]

    tail_rank = float(k)
    rank_score = np.full(
        (n, n),
        1.0 / np.sqrt(1.0 + tail_rank),
        dtype=float,
    )

    rows = np.arange(n)[:, None]
    rank_score[rows, top_indices] = 1.0 / np.sqrt(
        1.0 + local_ranks
    )

    mutual_rank = np.sqrt(rank_score * rank_score.T)

    alternative_scale = np.sqrt(
        alternatives[:, None] * alternatives[None, :]
    )
    margin = alternative_scale / (
        work + alternative_scale + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    density = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) / (
        np.sqrt(
            alternatives[:, None] * alternatives[None, :]
        )
        + scale_floor
    )
    density_factor = 0.95 + 0.05 * np.clip(density, 0.0, 1.0)

    score = (
        distance_affinity
        * endpoint_compatibility
        * mutual_rank
        * (1.0 + 0.08 * margin)
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
