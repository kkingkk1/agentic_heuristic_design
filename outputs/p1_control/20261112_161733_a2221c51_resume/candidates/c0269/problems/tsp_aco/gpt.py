import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-structural edge-priority scores for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (d + d.T)
    symmetric = np.maximum(symmetric, 0.0)

    valid_values = symmetric[valid]
    positive_values = valid_values[valid_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        scale = max(float(np.median(positive_values)), eps)
    else:
        scale = 1.0

    safe_distance = np.where(valid, symmetric, 0.0)
    positive_safe = np.maximum(safe_distance, eps)

    # Convert distances into a dense weighted graph over valid edges.
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(positive_safe[valid] / scale, 60.0)
    )

    # Normalize incident affinity so highly connected nodes do not dominate.
    degree = np.sum(affinity, axis=1)
    degree_safe = np.maximum(degree, eps)
    normalized_affinity = affinity / np.sqrt(
        degree_safe[:, None] * degree_safe[None, :]
    )
    normalized_affinity[~valid] = 0.0

    # Weighted graph Laplacian and effective-resistance proxy.
    laplacian = np.diag(np.sum(affinity, axis=1)) - affinity
    laplacian = 0.5 * (laplacian + laplacian.T)

    try:
        laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    except TypeError:
        laplacian_pinv = np.linalg.pinv(laplacian)

    resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance[~valid] = 0.0

    finite_resistance = resistance[valid & np.isfinite(resistance)]
    if finite_resistance.size:
        resistance_scale = max(
            float(np.median(finite_resistance)),
            eps,
        )
    else:
        resistance_scale = 1.0

    structural_importance = resistance / resistance_scale
    structural_importance = np.minimum(structural_importance, 20.0)

    # Preserve short-edge preference while rewarding structurally important links.
    distance_preference = np.exp(
        -np.minimum(positive_safe / scale, 60.0)
    )
    distance_preference[~valid] = 0.0

    bridge_reward = 1.0 + 0.55 * np.sqrt(structural_importance)
    bridge_reward[~valid] = 0.0

    score = (
        distance_preference
        * (0.35 + 0.65 * normalized_affinity)
        * bridge_reward
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
