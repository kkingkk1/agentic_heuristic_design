import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(d)
    if not np.all(finite | diagonal):
        raise ValueError(
            "distance_matrix contains invalid non-finite off-diagonal values"
        )

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(1.0, fallback)

    k = max(1, min(n - 1, int(np.sqrt(n))))
    candidate_indices = np.argpartition(work, kth=k - 1, axis=1)[:, :k]
    candidate_distances = np.take_along_axis(work, candidate_indices, axis=1)

    local_scale = np.median(candidate_distances, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, eps)

    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate_mask[rows, candidate_indices] = True
    np.fill_diagonal(candidate_mask, False)

    mutual = candidate_mask & candidate_mask.T
    one_sided = candidate_mask | candidate_mask.T

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )
    normalized_distance = work / (pair_scale + eps)

    locality = np.exp(-np.minimum(normalized_distance, 50.0))
    preference = np.where(
        mutual,
        1.0,
        np.where(one_sided, 0.62, 0.0),
    )

    score = locality * preference
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
