import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return scale-invariant, endpoint-aware TSP edge priorities."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & np.isfinite(distances.T)
        & (distances.T >= 0.0)
    )
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distances = np.where(valid, 0.5 * (distances + distances.T), np.inf)

    positive = symmetric_distances[valid & (symmetric_distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale_floor = max(float(np.median(positive)) * 1e-12, np.finfo(float).eps)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    fallback = max(fallback, scale_floor)

    d = np.where(valid, symmetric_distances, fallback)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    ordered = np.sort(d, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, scale_floor)

    if n > 2:
        alternative = ordered[:, 1]
    else:
        alternative = ordered[:, 0]
    alternative = np.maximum(alternative, scale_floor)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_length = d / (pair_scale + scale_floor)

    length_affinity = 1.0 / np.power(1.0 + normalized_length, 1.6)

    endpoint_i = 1.0 / (
        1.0 + d / (local_scale[:, None] + scale_floor)
    )
    endpoint_j = 1.0 / (
        1.0 + d / (local_scale[None, :] + scale_floor)
    )
    endpoint_affinity = np.sqrt(endpoint_i * endpoint_j)

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))
    rank_affinity_i = np.exp(-ranks / rank_scale)
    rank_affinity_j = rank_affinity_i.T
    mutual_rank = np.sqrt(rank_affinity_i * rank_affinity_j)

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = (alternative_scale - d) / (
        alternative_scale + d + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    competitive_gate = np.sqrt(
        np.clip(
            mutual_rank / (mutual_rank + 0.25),
            0.0,
            1.0,
        )
    )
    regret_bonus = 1.0 + 0.18 * margin * competitive_gate

    score = (
        length_affinity
        * endpoint_affinity
        * mutual_rank
        * regret_bonus
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
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
