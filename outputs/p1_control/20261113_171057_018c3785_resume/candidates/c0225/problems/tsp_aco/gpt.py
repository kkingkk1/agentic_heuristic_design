import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = valid & (d > 0.0)
    positive_values = d[positive]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    positive_distances = np.where(positive, d, np.inf)
    nearest_positive = np.min(positive_distances, axis=1)
    local_scale = np.where(
        np.isfinite(nearest_positive),
        nearest_positive,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    geometric_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            np.finfo(float).eps,
        )
    )

    global_affinity = np.divide(
        global_scale,
        global_scale + d,
        out=np.zeros_like(d, dtype=float),
        where=valid,
    )

    local_affinity = np.divide(
        geometric_scale,
        geometric_scale + d,
        out=np.zeros_like(d, dtype=float),
        where=valid,
    )

    order = np.argsort(
        np.where(valid, d, np.inf),
        axis=1,
        kind="stable",
    )

    ranks = np.empty_like(order)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n)[None, :]
    ranks = np.where(valid, ranks, n)

    rank_affinity = np.divide(
        1.0,
        1.0 + np.sqrt(ranks.astype(float)),
        out=np.zeros_like(d, dtype=float),
        where=valid,
    )

    candidate_width = min(n - 1, max(4, int(np.ceil(np.sqrt(n)))))
    candidate = valid & (ranks < candidate_width)

    reciprocal_rank = np.divide(
        1.0,
        1.0 + ranks.T.astype(float),
        out=np.zeros_like(d, dtype=float),
        where=valid.T,
    )

    mutual_knn = (
        candidate
        & candidate.T
        & valid
        & valid.T
    ).astype(float)

    nearest_mutual = (
        (ranks == 0)
        & (ranks.T == 0)
        & valid
        & valid.T
    ).astype(float)

    shared_candidate_count = (
        candidate.astype(float) @ candidate.astype(float).T
    )
    shared_candidate = np.divide(
        shared_candidate_count,
        float(max(candidate_width, 1)),
        out=np.zeros_like(d, dtype=float),
    )
    shared_candidate *= valid

    score = (
        0.28 * global_affinity
        + 0.38 * local_affinity
        + 0.14 * rank_affinity
        + 0.09 * reciprocal_rank
        + 0.07 * mutual_knn
        + 0.10 * nearest_mutual
        + 0.04 * shared_candidate
    )

    score = np.where(
        candidate,
        score,
        0.72 * score,
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
