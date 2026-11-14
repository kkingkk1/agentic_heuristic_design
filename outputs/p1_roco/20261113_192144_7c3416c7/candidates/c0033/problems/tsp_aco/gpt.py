import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a topology-aware edge-priority matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    valid_edge = finite | finite.T

    nonnegative = np.where(finite, np.maximum(d, 0.0), np.nan)
    symmetric = np.full((n, n), np.nan, dtype=float)

    both = finite & finite.T
    symmetric[both] = 0.5 * (nonnegative[both] + nonnegative.T[both])

    only_forward = finite & ~finite.T
    only_backward = ~finite & finite.T
    symmetric[only_forward] = nonnegative[only_forward]
    symmetric[only_backward] = nonnegative.T[only_backward]

    off_diagonal = ~np.eye(n, dtype=bool)
    observed = symmetric[off_diagonal & np.isfinite(symmetric)]

    if observed.size == 0:
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    deviation = float(np.median(np.abs(observed - np.median(observed))))
    fallback = np.median(observed) + 8.0 * max(deviation, scale)

    clean = np.where(np.isfinite(symmetric), symmetric, fallback)
    clean = np.maximum(clean, 0.0)
    np.fill_diagonal(clean, fallback)

    temperature = max(scale + deviation, np.finfo(float).eps)

    affinity = np.exp(-np.minimum(clean / temperature, 60.0))
    affinity[~valid_edge] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_mass = affinity.sum(axis=1, keepdims=True)
    transition = affinity / np.maximum(row_mass, np.finfo(float).eps)

    two_step_affinity = affinity @ affinity
    two_step_flow = transition @ transition

    # Participation in directed three-cycles: i -> j -> k -> i.
    triangle_support = affinity * two_step_affinity.T

    # Reciprocal two-step connectivity between the endpoints.
    reciprocal_support = np.sqrt(
        np.maximum(two_step_flow, 0.0)
        * np.maximum(two_step_flow.T, 0.0)
    )

    # Mutual one-step flow captures endpoint compatibility without using
    # a direct inverse-distance or rank formula.
    mutual_flow = np.sqrt(
        np.maximum(transition, 0.0)
        * np.maximum(transition.T, 0.0)
    )

    def normalize(matrix: np.ndarray) -> np.ndarray:
        values = matrix[off_diagonal & valid_edge]
        maximum = float(np.max(values)) if values.size else 0.0
        if not np.isfinite(maximum) or maximum <= 0.0:
            return np.zeros_like(matrix, dtype=float)
        return matrix / maximum

    score = (
        0.50 * normalize(triangle_support)
        + 0.30 * normalize(reciprocal_support)
        + 0.20 * normalize(mutual_flow)
    )

    score = 0.5 * (score + score.T)
    score[~valid_edge] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
