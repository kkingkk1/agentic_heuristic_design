import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priorities for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.percentile(positive, 75.0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 0.0)

    work = repaired.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps

    affinity = np.exp(
        -np.clip(work / max(scale, eps), 0.0, 50.0)
    )
    affinity[~np.isfinite(affinity)] = 0.0
    np.fill_diagonal(affinity, 0.0)

    strength = np.sum(affinity, axis=1)
    strength = np.maximum(strength, eps)

    endpoint_exclusivity = affinity / np.sqrt(
        strength[:, None] * strength[None, :]
    )

    mean_strength = float(np.mean(strength))
    if not np.isfinite(mean_strength) or mean_strength <= 0.0:
        mean_strength = 1.0

    exclusivity_factor = 0.65 + 0.35 * np.clip(
        endpoint_exclusivity / mean_strength,
        0.0,
        1.0,
    )

    common_neighbor_mass = affinity @ affinity
    normalized_common_mass = common_neighbor_mass / np.sqrt(
        strength[:, None] * strength[None, :]
    )

    bridge_factor = 1.0 / (
        1.0 + 1.5 * np.maximum(normalized_common_mass, 0.0)
    )
    bridge_factor = 0.82 + 0.18 * bridge_factor

    direct_factor = np.power(np.clip(affinity, 0.0, 1.0), 1.35)

    score = (
        direct_factor
        * exclusivity_factor
        * bridge_factor
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
