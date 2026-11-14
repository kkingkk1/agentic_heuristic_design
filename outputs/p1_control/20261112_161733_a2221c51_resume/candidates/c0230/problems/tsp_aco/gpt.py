import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw) & (raw >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (raw + raw.T)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.maximum(d, 0.0)

    eps = np.finfo(float).eps
    positive_mask = valid & (d > 0.0)
    positive = d[positive_mask]

    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)

    # Row-wise second-positive-neighbor scale without a full row sort.
    positive_work = np.where(positive_mask, d, np.inf)
    kth = min(1, n - 1)
    local_scale = np.partition(positive_work, kth=kth, axis=1)[:, kth]
    local_scale = np.where(np.isfinite(local_scale), local_scale, global_scale)
    local_scale = np.maximum(local_scale, eps)

    relative_length = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = np.where(valid, np.minimum(relative_length, 1e6), np.inf)

    # One stable sort supplies normalized endpoint ranks.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks /= max(n - 1, 1)
    ranks = np.where(valid, ranks, 1.0)

    endpoint_rank_score = np.sqrt(
        np.maximum(1.0 - ranks, eps)
        * np.maximum(1.0 - ranks.T, eps)
    )

    reciprocal_term = np.minimum(
        global_scale / np.maximum(d, eps),
        1e12,
    )

    denominator = max(n - 1, 1)
    nearest_k = min(3, denominator)
    mutual_nearest = (
        valid
        & (ranks <= nearest_k / denominator)
        & (ranks.T <= nearest_k / denominator)
    )

    local_neighbor_mask = (
        valid
        & (
            work
            <= 1.5 * np.minimum(local_scale[:, None], local_scale[None, :])
        )
    )
    local_density = local_neighbor_mask.mean(axis=1)

    congestion_penalty = (
        local_density[:, None] + local_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    log_score = (
        0.90 * np.log(np.maximum(reciprocal_term, eps))
        + 0.80 * np.log(np.maximum(endpoint_rank_score, eps))
        - relative_length
        - 0.25 * d / global_scale
        - 0.12 * congestion_penalty
        + 0.16 * mutual_nearest.astype(float)
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score = np.where(valid, score, 0.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
