import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_offdiag = np.isfinite(distances)
    np.fill_diagonal(finite_offdiag, False)

    positive = distances[finite_offdiag & (distances > 0.0)]
    if positive.size:
        fallback = float(np.median(positive))
    elif np.any(finite_offdiag):
        fallback = max(float(np.median(distances[finite_offdiag])), 1.0)
    else:
        fallback = 1.0

    fallback = max(fallback, np.finfo(float).eps)

    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    scale = float(np.median(d[~np.eye(n, dtype=bool)]))
    scale = max(scale, np.finfo(float).eps)

    profile = d / scale
    profile_diff = np.abs(profile[:, None, :] - profile[None, :, :])

    mask = ~np.eye(n, dtype=bool)
    profile_diff[:, :, np.arange(n)] = 0.0

    profile_distance = np.sum(profile_diff, axis=2)
    profile_distance /= max(n - 2, 1)

    profile_similarity = np.exp(-np.clip(profile_distance, 0.0, 40.0))
    profile_similarity[~mask] = 0.0

    two_hop = d[:, :, None] + d[None, :, :]
    mediator_mask = np.ones((n, n, n), dtype=bool)
    indices = np.arange(n)
    mediator_mask[indices, indices, :] = False
    mediator_mask[indices, :, indices] = False
    mediator_mask[:, indices, indices] = False
    two_hop[~mediator_mask] = np.inf

    best_detour = np.min(two_hop, axis=2)
    direct = np.maximum(d, np.finfo(float).eps)

    detour_leverage = np.zeros((n, n), dtype=float)
    valid_detour = np.isfinite(best_detour)
    detour_leverage[valid_detour] = np.maximum(
        best_detour[valid_detour] - direct[valid_detour],
        0.0,
    )
    detour_leverage /= best_detour + direct + np.finfo(float).eps
    detour_leverage[~mask] = 0.0

    common_support = np.sum(
        np.exp(-np.clip(two_hop / scale, 0.0, 40.0))
        * mediator_mask,
        axis=2,
    )
    common_support /= max(n - 2, 1)
    common_support[~mask] = 0.0

    score = (
        0.50 * profile_similarity
        + 0.35 * detour_leverage
        + 0.15 * common_support
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
