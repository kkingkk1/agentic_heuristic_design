import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-priority indicators for stochastic TSP solution sampling.
    Larger values indicate more promising directed edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & off_diagonal

    if not np.any(valid):
        return result

    positive = d[np.isfinite(d) & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    large_distance = scale * 1e6
    safe_d = np.where(valid, d, large_distance)
    np.fill_diagonal(safe_d, large_distance)

    def normalize(signal: np.ndarray) -> np.ndarray:
        values = signal[valid]
        if values.size == 0:
            return np.zeros_like(signal)
        minimum = np.min(values)
        maximum = np.max(values)
        if maximum <= minimum:
            normalized = np.zeros_like(signal)
            normalized[valid] = 1.0 if maximum > 0.0 else 0.0
            return normalized
        normalized = np.zeros_like(signal)
        normalized[valid] = (values - minimum) / (maximum - minimum)
        return normalized

    inverse_distance = np.zeros_like(safe_d)
    inverse_distance[valid] = 1.0 / (safe_d[valid] + np.finfo(float).eps)
    inverse_distance = normalize(inverse_distance)

    reciprocal = normalize(
        0.5 * (inverse_distance + inverse_distance.T)
    )

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros_like(safe_d)
    rows = np.arange(n)
    mutual[rows, nearest] = (
        valid[rows, nearest] & (nearest[nearest] == rows)
    ).astype(float)
    mutual = normalize(mutual)

    valid_count = np.sum(valid, axis=1)
    sorted_distances = np.sort(safe_d, axis=1)

    nearest_distance = sorted_distances[:, 0]
    alternative_distance = np.where(
        valid_count >= 2,
        sorted_distances[:, 1],
        np.inf,
    )

    savings = (
        alternative_distance[:, None]
        + alternative_distance[None, :]
        - safe_d
    )
    savings = np.maximum(savings, 0.0)

    savings_valid = (
        valid
        & np.isfinite(alternative_distance[:, None])
        & np.isfinite(alternative_distance[None, :])
    )
    savings[~savings_valid] = 0.0
    savings = normalize(savings)

    score = (
        0.50 * inverse_distance
        + 0.20 * reciprocal
        + 0.15 * mutual
        + 0.15 * savings
    )

    score[~valid] = 0.0

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
