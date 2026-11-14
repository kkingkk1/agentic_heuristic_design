import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric reciprocal-affinity indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)
    symmetric_valid = valid & valid.T

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    floor = max(scale * 1e-12, np.finfo(float).eps)
    fallback = float(np.percentile(positive, 95.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = scale

    d = np.where(valid, distances, fallback)
    d = np.maximum(0.5 * (d + d.T), 0.0)
    np.fill_diagonal(d, np.inf)

    affinity = 1.0 / (1.0 + d / (scale + floor))
    affinity = np.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)
    affinity[~valid] = 0.0

    row_mass = np.sum(affinity, axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, floor)
    preference = affinity / row_mass

    mutual_preference = np.sqrt(
        np.maximum(preference * preference.T, 0.0)
    )

    exclusivity = np.sqrt(
        np.maximum(
            (preference / (np.max(preference, axis=1, keepdims=True) + floor))
            * (
                preference.T
                / (np.max(preference, axis=1, keepdims=True).T + floor)
            ),
            0.0,
        )
    )

    normalized = affinity / (np.max(affinity, axis=1, keepdims=True) + floor)
    ambiguity = np.sum(normalized * normalized, axis=1)
    ambiguity = ambiguity / (np.max(ambiguity) + floor)
    ambiguity_factor = np.sqrt(
        np.clip((1.0 - ambiguity[:, None]) * (1.0 - ambiguity[None, :]), 0.0, 1.0)
    )

    score = (
        mutual_preference
        * (0.35 + 0.65 * exclusivity)
        * (0.70 + 0.30 * ambiguity_factor)
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
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
