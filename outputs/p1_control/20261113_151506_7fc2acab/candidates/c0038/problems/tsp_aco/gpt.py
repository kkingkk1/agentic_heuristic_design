import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a globally informed spectral edge-priority matrix for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    positive = d[off_diagonal & (d > 0)]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    # Convert distances into positive graph conductances.
    conductance = np.exp(-d / scale)
    np.fill_diagonal(conductance, 0.0)
    conductance = 0.5 * (conductance + conductance.T)

    # Weighted graph Laplacian.
    laplacian = np.diag(np.sum(conductance, axis=1)) - conductance
    laplacian = 0.5 * (laplacian + laplacian.T)

    # Moore-Penrose inverse through a symmetric eigendecomposition.
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    tolerance = max(
        np.finfo(float).eps * n * max(float(np.max(eigenvalues)), 1.0),
        np.finfo(float).eps,
    )

    inverse_values = np.zeros_like(eigenvalues)
    nonzero = eigenvalues > tolerance
    inverse_values[nonzero] = 1.0 / eigenvalues[nonzero]

    laplacian_pinv = (
        eigenvectors * inverse_values[None, :]
    ) @ eigenvectors.T
    laplacian_pinv = 0.5 * (laplacian_pinv + laplacian_pinv.T)

    # Effective resistance for every pair of nodes.
    diagonal = np.diag(laplacian_pinv)
    resistance = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance = 0.5 * (resistance + resistance.T)
    np.fill_diagonal(resistance, 0.0)

    # Electrical leverage identifies globally important edges.
    leverage = conductance * resistance

    # Retain a mild direct-distance preference.
    distance_preference = np.exp(-d / scale)
    score = leverage * (0.70 + 0.30 * distance_preference)

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
