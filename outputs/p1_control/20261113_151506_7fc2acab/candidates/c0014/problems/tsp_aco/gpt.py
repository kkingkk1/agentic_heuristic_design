import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = distances.copy()
    d[~np.isfinite(d)] = np.inf
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    proximity = np.exp(-d / scale)
    proximity[~np.isfinite(proximity)] = 0.0
    np.fill_diagonal(proximity, 0.0)

    row_max = np.max(proximity, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    local_preference = proximity / row_max

    compatibility = local_preference @ local_preference.T
    compatibility /= max(n - 2, 1)
    np.fill_diagonal(compatibility, 0.0)

    endpoint_strength = local_preference.sum(axis=1, keepdims=True)
    endpoint_strength[endpoint_strength <= 0] = 1.0

    competition = (
        local_preference / endpoint_strength
        + local_preference.T / endpoint_strength.T
    )
    competition = np.clip(competition, 0.0, 1.0)

    score = (
        0.60 * local_preference
        + 0.25 * compatibility
        + 0.15 * (1.0 - competition)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)


if __name__ == "__main__":
    example = np.array(
        [
            [0.0, 2.0, 5.0, 4.0],
            [2.0, 0.0, 3.0, 6.0],
            [5.0, 3.0, 0.0, 1.0],
            [4.0, 6.0, 1.0, 0.0],
        ]
    )
    print(heuristics_v2(example))
