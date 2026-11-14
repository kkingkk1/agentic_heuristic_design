import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric spectral edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    distances = 0.5 * (distances + distances.T)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe_distances = np.where(valid, distances, 0.0)

    # Convert distances to bounded conductances.
    conductance = np.zeros((n, n), dtype=float)
    conductance[valid] = np.exp(
        -np.minimum(safe_distances[valid] / scale, 50.0)
    )

    # Build the weighted graph Laplacian.
    degree = np.sum(conductance, axis=1)
    laplacian = np.diag(degree) - conductance

    # The pseudoinverse also handles disconnected or degenerate valid graphs.
    laplacian_pinv = np.linalg.pinv(laplacian, rcond=1e-10)
    laplacian_pinv = 0.5 * (laplacian_pinv + laplacian_pinv.T)

    diagonal = np.diag(laplacian_pinv)
    effective_resistance = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2.0 * laplacian_pinv
    )
    effective_resistance = np.maximum(effective_resistance, 0.0)

    # Spectral leverage favors edges that are both usable and structurally informative.
    leverage = conductance * effective_resistance
    short_edge_factor = np.zeros((n, n), dtype=float)
    short_edge_factor[valid] = np.exp(
        -0.25 * np.minimum(safe_distances[valid] / scale, 50.0)
    )

    score = np.power(np.maximum(leverage, 0.0), 0.75) * short_edge_factor
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
