import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return sparse, endpoint-aware edge-priority indicators for TSP sampling."""
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

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    symmetric_valid = valid & valid.T
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    positive_scale = max(float(np.median(positive)), eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    order = np.argsort(d, axis=1, kind="mergesort")
    nearest = order[:, :k]

    local_scale = np.take_along_axis(d, nearest, axis=1).mean(axis=1)
    local_scale = np.maximum(local_scale, positive_scale * 1e-10)

    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = d / (pair_scale + eps)
    length_score = 1.0 / (1.0 + relative_length)

    if n > 2:
        second_nearest = d[order[:, 1]]
    else:
        second_nearest = d[order[:, 0]]

    alternative_scale = np.sqrt(
        second_nearest[:, None] * second_nearest[None, :]
    )
    detour_margin = alternative_scale / (
        d + alternative_scale + eps
    )
    detour_margin = np.clip(detour_margin, 0.0, 1.0)

    score = (
        length_score
        * (0.65 + 0.35 * mutual_rank)
        * (1.0 + 0.10 * detour_margin)
    )

    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_mask[rows, nearest] = True
    candidate_mask |= candidate_mask.T
    candidate_mask &= symmetric_valid

    score *= np.where(candidate_mask, 1.0, 0.08)
    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
