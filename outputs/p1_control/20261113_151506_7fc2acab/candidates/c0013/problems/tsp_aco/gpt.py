import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return symmetric edge-priority indicators for stochastic TSP solution sampling.

    High values indicate edges that are simultaneously attractive and locally
    scarce alternatives for both of their endpoints.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0.0)
    valid = positive & ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    safe_d = np.where(valid, d, np.inf)

    # Robust global distance scale.
    positive_values = safe_d[np.isfinite(safe_d)]
    scale = float(np.median(positive_values))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    # Convert distances to smooth local preferences.
    attractiveness = np.zeros((n, n), dtype=float)
    attractiveness[valid] = np.exp(
        -np.minimum(safe_d[valid] / scale, 50.0)
    )

    # Normalize each row so scores express competition among neighbors.
    row_sum = attractiveness.sum(axis=1, keepdims=True)
    row_preference = np.divide(
        attractiveness,
        row_sum,
        out=np.zeros_like(attractiveness),
        where=row_sum > 0.0,
    )

    # Local scarcity: edges near the best neighbor receive a stronger bonus.
    row_max = row_preference.max(axis=1, keepdims=True)
    scarcity = np.divide(
        row_preference,
        row_max,
        out=np.zeros_like(row_preference),
        where=row_max > 0.0,
    )

    # Require support from both endpoints.
    mutual_support = np.sqrt(
        np.maximum(scarcity, 0.0) * np.maximum(scarcity.T, 0.0)
    )

    # Add a bounded global-distance component for disconnected local patterns.
    global_component = np.zeros_like(d, dtype=float)
    global_component[valid] = 1.0 / (1.0 + d[valid] / scale)

    score = (
        0.60 * mutual_support
        + 0.25 * np.sqrt(
            np.maximum(row_preference, 0.0)
            * np.maximum(row_preference.T, 0.0)
        )
        + 0.15 * (global_component + global_component.T) / 2.0
    )

    score = (score + score.T) / 2.0
    score[~(valid | valid.T)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
