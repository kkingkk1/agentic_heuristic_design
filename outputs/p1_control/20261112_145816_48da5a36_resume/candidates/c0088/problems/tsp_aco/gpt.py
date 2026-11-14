import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        finite_values = distances[finite]
        nonnegative = finite_values[finite_values >= 0]
        replacement = float(np.max(nonnegative)) if nonnegative.size else 1.0
        distances = np.where(finite, distances, replacement)

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    positive = distances[np.isfinite(distances) & (distances > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    safe_distances = np.where(np.isfinite(distances), distances, global_scale)

    inverse_distance = 1.0 / np.maximum(safe_distances, np.finfo(float).eps)
    np.fill_diagonal(inverse_distance, 0.0)
    inverse_max = float(np.max(inverse_distance))
    if inverse_max > 0:
        inverse_distance /= inverse_max

    sorted_distances = np.sort(safe_distances, axis=1)
    nearest_count = max(1, min(3, n - 1))
    local_scale = np.mean(sorted_distances[:, :nearest_count], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative_proximity = np.exp(
        -safe_distances / np.sqrt(np.outer(local_scale, local_scale))
    )
    np.fill_diagonal(relative_proximity, 0.0)

    order = np.argsort(safe_distances, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    rank_score = 1.0 / (1.0 + ranks.astype(float))
    rank_score[:, np.arange(n)] = 0.0
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    mutual_preference = np.sqrt(rank_score * rank_score.T)
    np.fill_diagonal(mutual_preference, 0.0)

    density = global_scale / np.maximum(local_scale, np.finfo(float).eps)
    density /= max(float(np.median(density)), np.finfo(float).eps)
    density = np.clip(density, 0.25, 4.0)

    harmonic_density = (
        2.0 * density[:, None] * density[None, :]
        / np.maximum(density[:, None] + density[None, :], np.finfo(float).eps)
    )
    density_contrast = np.minimum(density[:, None], density[None, :]) / np.maximum(
        density[:, None], density[None, :]
    )

    bridge_floor = inverse_distance * (
        0.35
        + 0.25 * density_contrast
        + 0.15 * np.minimum(harmonic_density, 1.0)
    )
    bridge_floor = np.minimum(bridge_floor, 0.45)

    endpoint_competition = np.sqrt(
        np.maximum(rank_score, 0.0) * np.maximum(rank_score.T, 0.0)
    )
    score = (
        0.42 * inverse_distance
        + 0.30 * relative_proximity
        + 0.18 * mutual_preference
        + 0.10 * bridge_floor
    )
    score *= 0.92 + 0.08 * endpoint_competition
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
