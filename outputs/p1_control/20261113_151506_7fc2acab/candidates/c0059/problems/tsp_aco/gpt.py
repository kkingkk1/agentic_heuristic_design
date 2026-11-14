import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    raw = distance_matrix.copy()
    finite = np.isfinite(raw)

    if finite.any():
        replacement = float(np.median(raw[finite]))
        raw[~finite] = replacement
    else:
        return np.zeros((n, n), dtype=float)

    raw = np.maximum(raw, 0.0)
    d = 0.5 * (raw + raw.T)
    np.fill_diagonal(d, np.inf)

    finite_edges = d[np.isfinite(d)]
    if finite_edges.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_edges))
    d[~np.isfinite(d)] = fallback

    positive = d[d > 0.0]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    nearest_k = max(1, min(5, n - 1))
    order = np.argsort(d, axis=1)

    nearest = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    nearest[rows, order[:, :nearest_k]] = 1.0

    mutual = nearest * nearest.T
    one_sided = 0.5 * (nearest + nearest.T)

    local_scale = np.partition(d, nearest_k - 1, axis=1)[:, nearest_k - 1]
    local_scale = np.maximum(local_scale, global_scale * 1e-6)

    endpoint_relative = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
        + global_scale * global_scale * 1e-12
    )

    local_closeness = np.exp(-np.maximum(endpoint_relative - 1.0, 0.0))
    local_closeness = np.clip(local_closeness, 0.0, 1.0)

    inverse_distance = 1.0 / (d + global_scale * 1e-6)
    inverse_distance /= max(float(np.max(inverse_distance)), np.finfo(float).eps)

    preference_i = np.exp(-d / (local_scale[:, None] + global_scale * 1e-6))
    preference_j = preference_i.T
    bilateral_preference = np.sqrt(preference_i * preference_j)

    score = (
        0.48 * inverse_distance
        + 0.24 * bilateral_preference
        + 0.14 * local_closeness
        + 0.09 * one_sided
        + 0.05 * mutual
    )

    np.fill_diagonal(score, 0.0)

    score_min = float(np.min(score))
    score_max = float(np.max(score))

    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score
