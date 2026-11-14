import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest_block = np.partition(work, k - 1, axis=1)[:, :k]
    kth_distance = np.max(nearest_block, axis=1)
    local_scale = np.maximum(nearest_block.mean(axis=1), eps)

    if k == 1:
        alternative_scale = np.maximum(kth_distance, eps)
    else:
        second_block = np.partition(work, 1, axis=1)[:, :2]
        alternative_scale = np.maximum(second_block[:, 1], eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = d / (pair_scale + eps)

    closeness = 1.0 / (1.0 + relative_length)

    endpoint_rank = 1.0 / (
        1.0
        + d / (local_scale[:, None] + eps)
        + d / (local_scale[None, :] + eps)
    )

    alternative_pair = alternative_scale[:, None] + alternative_scale[None, :]
    detour_margin = alternative_pair / (d + alternative_pair + eps)

    row_candidates = d <= (kth_distance[:, None] + eps)
    col_candidates = row_candidates.T
    candidate_weight = np.where(row_candidates | col_candidates, 1.0, 0.35)

    mutual_candidate = row_candidates & col_candidates
    candidate_weight *= np.where(mutual_candidate, 1.08, 1.0)

    score = (
        closeness
        * (0.55 + 0.45 * endpoint_rank)
        * (0.70 + 0.30 * detour_margin)
        * candidate_weight
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score = score / maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
