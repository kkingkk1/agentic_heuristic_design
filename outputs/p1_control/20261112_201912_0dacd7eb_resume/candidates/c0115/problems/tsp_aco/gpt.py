import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    raw_valid = np.isfinite(distances) & (distances > 0)
    raw_valid &= ~np.eye(n, dtype=bool)

    positive = distances[raw_valid]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    forward = np.where(raw_valid, distances, fallback)
    reverse = forward.T

    both_valid = raw_valid & raw_valid.T
    one_valid = raw_valid | raw_valid.T

    work = np.where(
        both_valid,
        0.5 * (forward + reverse),
        np.where(raw_valid, forward, reverse),
    )
    work = np.where(one_valid, work, fallback)
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    valid = np.isfinite(work) & (work > 0)
    valid &= ~np.eye(n, dtype=bool)

    positive_work = work[valid]
    if positive_work.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive_work))
    safe = np.where(valid, work, fallback)
    np.fill_diagonal(safe, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest_values = np.partition(safe, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest_values), eps)
    node_scale = np.maximum(
        np.mean(nearest_values, axis=1),
        eps,
    )

    pair_scale = np.sqrt(node_scale[:, None] * node_scale[None, :])
    normalized_length = safe / (pair_scale + eps)

    closeness = np.exp(-np.minimum(normalized_length, 50.0))
    closeness[~valid] = 0.0

    order = np.argsort(safe, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~valid] = 0.0

    reciprocal_rank = np.sqrt(rank_score * rank_score.T)
    reciprocal_rank[~valid] = 0.0

    nearest = np.argmin(safe, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if i != j and valid[i, j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    alternatives = safe[:, :, None] + safe[None, :, :]
    alternatives -= safe[:, None, :]
    alternatives = np.transpose(alternatives, (0, 2, 1))

    alternatives = (
        safe[:, :, None] + safe[None, :, :]
    )

    invalid_intermediates = (
        ~np.isfinite(safe)[:, None, :]
        | ~np.isfinite(safe)[None, :, :]
    )
    alternatives[invalid_intermediates] = np.inf

    alternatives[
        np.eye(n, dtype=bool)[:, :, None]
        | np.eye(n, dtype=bool)[None, :, :]
    ] = np.inf

    replacement_cost = np.min(alternatives, axis=2)
    replacement_gap = np.maximum(replacement_cost - safe, 0.0)
    replacement_gap[~valid] = 0.0

    indispensability = replacement_gap / (
        replacement_gap + pair_scale + eps
    )
    indispensability[~valid] = 0.0

    density_bonus = np.exp(
        -np.minimum(
            normalized_length
            / (np.sqrt(node_scale[:, None] * node_scale[None, :]) + eps),
            50.0,
        )
    )
    density_bonus[~valid] = 0.0

    score = (
        0.48 * closeness
        + 0.22 * reciprocal_rank
        + 0.14 * mutual_nearest
        + 0.12 * indispensability
        + 0.04 * density_bonus
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
