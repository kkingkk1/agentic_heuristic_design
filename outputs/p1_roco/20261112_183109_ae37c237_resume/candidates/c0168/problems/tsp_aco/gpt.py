import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros_like(distances, dtype=float)

    d = 0.5 * (distances + distances.T)
    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    eps = np.finfo(float).eps

    global_scale = float(np.median(positive))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = float(np.mean(positive))
    global_scale = max(global_scale, eps)

    fallback = float(np.max(positive))
    working = np.where(valid, d, fallback)
    np.fill_diagonal(working, fallback)

    k = min(7, n - 1)
    nearest = np.argpartition(working, kth=k - 1, axis=1)[:, :k]
    nearest_values = np.take_along_axis(working, nearest, axis=1)

    local_scale = np.median(nearest_values, axis=1)
    local_scale = np.maximum(local_scale, eps)

    relative = d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = np.maximum(relative, eps)

    robust_distance = np.log1p(relative) / np.log1p(
        np.maximum(global_scale / np.median(local_scale), 1.0)
    )
    robust_distance = np.maximum(robust_distance, eps)

    distance_term = np.exp(-1.15 * np.minimum(robust_distance, 8.0))
    distance_term *= np.exp(
        -0.35 * np.maximum(np.log1p(relative) - 1.0, 0.0)
    )

    order = np.argsort(working, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_decay = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_preference = np.exp(-ranks / rank_decay)
    rank_preference *= valid

    endpoint_preference = np.sqrt(
        rank_preference * rank_preference.T
    )

    candidate = np.zeros((n, n), dtype=float)
    candidate[np.arange(n)[:, None], nearest] = 1.0
    candidate *= valid

    mutual_candidate = candidate * candidate.T
    shared_candidate = (candidate @ candidate.T) / float(k)
    shared_candidate = np.clip(shared_candidate, 0.0, 1.0)

    nearest_bonus = 1.0 + 0.55 * mutual_candidate
    shared_bonus = 1.0 + 0.20 * shared_candidate

    score = (
        distance_term
        * (0.35 + 0.65 * endpoint_preference)
        * nearest_bonus
        * shared_bonus
    )
    score *= valid

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    score = 0.5 * (score + score.T)
    score *= valid
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
