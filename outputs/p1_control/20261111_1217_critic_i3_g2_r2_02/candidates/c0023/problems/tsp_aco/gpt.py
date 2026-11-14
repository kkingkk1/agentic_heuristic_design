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

    positive = work[np.isfinite(work) & (work > 0.0)]
    global_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    eps = np.finfo(float).eps * max(1.0, global_scale)

    k = max(1, min(n - 1, int(np.sqrt(n))))

    candidate_idx = np.argpartition(work, k - 1, axis=1)[:, :k]
    candidate_dist = np.take_along_axis(work, candidate_idx, axis=1)

    local_scale = np.median(candidate_dist, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        np.maximum(local_scale, eps),
        max(global_scale, eps),
    )

    local_rank_order = np.argsort(candidate_dist, axis=1, kind="stable")
    local_rank = np.empty_like(local_rank_order, dtype=float)
    row_ids = np.arange(n)[:, None]
    local_rank[row_ids, local_rank_order] = np.arange(k, dtype=float)[None, :]
    local_rank /= max(1.0, float(k - 1))

    rows = np.repeat(np.arange(n), k)
    cols = candidate_idx.reshape(-1)
    values = candidate_dist.reshape(-1)
    ranks = local_rank.reshape(-1)

    valid = np.isfinite(values) & (rows != cols)

    score = np.zeros((n, n), dtype=float)

    if np.any(valid):
        r = rows[valid]
        c = cols[valid]
        edge_dist = values[valid]
        edge_rank = ranks[valid]

        pair_scale = np.sqrt(
            local_scale[r] * local_scale[c]
        )
        normalized_local = edge_dist / (pair_scale + eps)

        local_attraction = np.exp(-normalized_local)
        rank_attraction = np.exp(-1.6 * edge_rank)

        raw_attraction = np.exp(
            -0.15 * edge_dist / (global_scale + eps)
        )

        directed_score = (
            local_attraction
            * (0.72 + 0.28 * rank_attraction)
            * (0.85 + 0.15 * raw_attraction)
        )

        score[r, c] = directed_score

    reverse_candidate = score.T > 0.0
    forward_candidate = score > 0.0

    score *= np.where(
        reverse_candidate,
        1.08,
        0.94,
    )

    score[~finite] = 0.0
    score[diagonal] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
