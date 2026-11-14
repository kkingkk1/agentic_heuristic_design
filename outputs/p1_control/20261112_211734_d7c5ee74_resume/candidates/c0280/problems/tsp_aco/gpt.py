import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
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

    work = np.where(valid, distances, fallback)
    work = np.maximum(work, 0.0)
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    finite_offdiag = work[np.isfinite(work)]
    replacement = (
        float(np.max(finite_offdiag))
        if finite_offdiag.size
        else fallback
    )
    if not np.isfinite(replacement) or replacement <= 0.0:
        replacement = fallback

    work = np.where(np.isfinite(work), work, replacement)
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    positive_work = work[np.isfinite(work) & (work > 0.0)]
    global_scale = (
        float(np.median(positive_work))
        if positive_work.size
        else replacement
    )
    global_scale = max(global_scale, eps)

    k = max(1, min(n - 1, int(np.rint(np.sqrt(n)))))

    partitioned = np.partition(work, kth=k - 1, axis=1)
    neighborhood = partitioned[:, :k]

    local_scale = np.maximum(
        np.mean(neighborhood, axis=1),
        global_scale * 1e-10,
    )

    if k < n - 1:
        alternative = partitioned[:, k]
    else:
        alternative = partitioned[:, k - 1]
    alternative = np.maximum(alternative, global_scale * 1e-10)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_length = work / np.maximum(pair_scale, eps)
    distance_affinity = np.exp(
        -np.clip(normalized_length ** 0.9, 0.0, 50.0)
    )

    endpoint_i = np.exp(
        -np.clip(
            (work / np.maximum(local_scale[:, None], eps)) ** 0.85,
            0.0,
            50.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            (work / np.maximum(local_scale[None, :], eps)) ** 0.85,
            0.0,
            50.0,
        )
    )

    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)
    bottleneck_affinity = np.minimum(endpoint_i, endpoint_j)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_width = float(max(k, 1))
    rank_i = np.exp(-np.clip(ranks / rank_width, 0.0, 50.0))
    rank_j = rank_i.T

    mutual_rank = np.sqrt(rank_i * rank_j)
    candidate_gate = 0.60 + 0.40 * mutual_rank

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = (
        alternative_scale - work
    ) / (
        alternative_scale + work + global_scale
    )
    margin = np.clip(margin, 0.0, 1.0)

    margin_bonus = 1.0 + 0.08 * margin * np.sqrt(
        np.clip(mutual_rank, 0.0, 1.0)
    )

    score = (
        distance_affinity
        * endpoint_compatibility
        * bottleneck_affinity
        * mutual_rank
        * candidate_gate
        * margin_bonus
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
