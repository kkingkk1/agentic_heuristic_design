import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distances, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[np.arange(n), np.arange(n)] = np.inf

    finite_rank = np.where(np.isfinite(ranks), ranks, 0.0)

    nearest = d[np.arange(n), order[:, 0]]
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    second_index = min(1, n - 2)
    third_index = min(2, n - 2)

    second_nearest = d[np.arange(n), order[:, second_index]]
    third_nearest = d[np.arange(n), order[:, third_index]]

    second_nearest = np.where(
        np.isfinite(second_nearest), second_nearest, nearest
    )
    third_nearest = np.where(
        np.isfinite(third_nearest), third_nearest, second_nearest
    )

    local_scale = 0.55 * second_nearest + 0.45 * third_nearest
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    distance_affinity = np.exp(-d / pair_scale)
    distance_affinity[~np.isfinite(distance_affinity)] = 0.0

    rank_affinity = 1.0 / (1.0 + finite_rank)
    rank_affinity[~np.isfinite(rank_affinity)] = 0.0
    rank_affinity[np.arange(n), np.arange(n)] = 0.0

    mutual_rank = np.sqrt(rank_affinity * rank_affinity.T)
    mutual_distance = np.sqrt(distance_affinity * distance_affinity.T)

    nearest_mask = np.zeros((n, n), dtype=float)
    nearest_mask[np.arange(n), order[:, 0]] = 1.0

    top_two_mask = (finite_rank <= 1.0).astype(float)
    np.fill_diagonal(nearest_mask, 0.0)
    np.fill_diagonal(top_two_mask, 0.0)

    mutual_nearest = nearest_mask * nearest_mask.T
    mutual_top_two = top_two_mask * top_two_mask.T

    third_rank_bonus = np.exp(-0.5 * np.maximum(finite_rank - 2.0, 0.0))
    third_rank_bonus *= (finite_rank <= 2.0).astype(float)
    third_rank_bonus[np.arange(n), np.arange(n)] = 0.0
    third_rank_bonus = 0.5 * (third_rank_bonus + third_rank_bonus.T)

    savings = nearest[:, None] + nearest[None, :] - d
    savings = np.maximum(savings, 0.0)
    positive_savings = savings[np.isfinite(savings) & (savings > 0)]
    savings_scale = (
        float(np.median(positive_savings))
        if positive_savings.size
        else global_scale
    )
    savings_scale = max(savings_scale, np.finfo(float).eps)
    savings_score = np.tanh(savings / savings_scale)

    score = (
        0.36 * mutual_distance
        + 0.27 * mutual_rank
        + 0.16 * distance_affinity
        + 0.10 * mutual_nearest
        + 0.06 * mutual_top_two
        + 0.03 * third_rank_bonus
        + 0.02 * savings_score
    )

    candidate_cutoff = min(4, n - 1)
    candidate_mask = (finite_rank <= candidate_cutoff) | (
        finite_rank.T <= candidate_cutoff
    )
    score *= np.where(candidate_mask, 1.0, 0.88)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
