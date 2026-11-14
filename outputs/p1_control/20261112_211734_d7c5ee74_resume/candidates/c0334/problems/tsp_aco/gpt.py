import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances >= 0.0) & off_diagonal
    symmetric_valid = valid & valid.T

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    d = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(d, np.inf)

    scale_floor = max(
        float(np.median(positive)) * 1e-10,
        np.finfo(float).eps,
    )

    k = max(
        1,
        min(n - 1, int(np.ceil(np.sqrt(n)))),
    )

    local_candidates = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.maximum(
        np.mean(local_candidates, axis=1),
        scale_floor,
    )

    if n > 2:
        alternative = np.partition(d, kth=1, axis=1)[:, 1]
    else:
        alternative = local_candidates[:, 0]

    alternative = np.where(
        np.isfinite(alternative),
        alternative,
        local_scale,
    )
    alternative = np.maximum(alternative, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / (pair_scale + scale_floor)

    distance_attraction = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 40.0)
    )

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = max(float(k), 1.0)
    rank_support = np.exp(
        -0.62 * (ranks + ranks.T) / rank_scale
    )

    endpoint_i = np.exp(
        -np.clip(
            0.88 * d / (local_scale[:, None] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            0.88 * d / (local_scale[None, :] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_competition = np.sqrt(endpoint_i * endpoint_j)

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_factor = 0.96 + 0.06 * margin

    candidate_i = np.exp(
        -np.clip(ranks / rank_scale, 0.0, 40.0)
    )
    candidate_j = candidate_i.T
    mutual_candidate_support = np.sqrt(candidate_i * candidate_j)
    candidate_gate = 0.58 + 0.42 * mutual_candidate_support

    neighborhood_affinity = np.exp(
        -np.clip(
            d / (local_scale[:, None] + scale_floor),
            0.0,
            40.0,
        )
    )
    np.fill_diagonal(neighborhood_affinity, 0.0)

    common_support = neighborhood_affinity @ neighborhood_affinity.T
    row_norm = np.sqrt(
        np.sum(neighborhood_affinity**2, axis=1)
    )
    denominator = (
        row_norm[:, None] * row_norm[None, :]
        + scale_floor
    )
    common_compatibility = np.clip(
        common_support / denominator,
        0.0,
        1.0,
    )
    compatibility_factor = 0.96 + 0.04 * common_compatibility

    score = (
        distance_attraction
        * rank_support
        * endpoint_competition
        * candidate_gate
        * margin_factor
        * compatibility_factor
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
