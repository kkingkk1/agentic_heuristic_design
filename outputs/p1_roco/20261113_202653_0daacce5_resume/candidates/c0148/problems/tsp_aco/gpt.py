import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global spectral edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    temperature = max(
        scale,
        np.sqrt(np.finfo(float).eps) * max(scale, 1.0),
    )

    conductance = np.zeros((n, n), dtype=float)
    conductance[valid] = np.exp(
        -np.minimum(distances[valid] / temperature, 60.0)
    )
    conductance = 0.5 * (conductance + conductance.T)
    conductance[~valid] = 0.0
    np.fill_diagonal(conductance, 0.0)

    degree = conductance.sum(axis=1)
    positive_degree = degree > 0.0

    laplacian = np.diag(degree) - conductance
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        tolerance = max(
            float(np.max(eigenvalues)) * n * np.finfo(float).eps,
            np.finfo(float).eps,
        )
        inverse_eigenvalues = np.zeros_like(eigenvalues)
        nonzero = eigenvalues > tolerance
        inverse_eigenvalues[nonzero] = 1.0 / eigenvalues[nonzero]
        laplacian_pinv = (
            eigenvectors
            @ (inverse_eigenvalues[:, None] * eigenvectors.T)
        )
    except np.linalg.LinAlgError:
        laplacian_pinv = np.linalg.pinv(laplacian)

    effective_resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    effective_resistance = np.maximum(effective_resistance, 0.0)

    tree_marginal = conductance * effective_resistance
    tree_marginal[~valid] = 0.0

    normalized_conductance = np.zeros_like(conductance)
    degree_scale = np.maximum(
        np.sqrt(degree[:, None] * degree[None, :]),
        np.finfo(float).eps,
    )
    normalized_conductance[valid] = (
        conductance[valid] / degree_scale[valid]
    )

    two_hop = normalized_conductance @ normalized_conductance
    two_hop[~valid] = 0.0
    np.fill_diagonal(two_hop, 0.0)

    def normalize(matrix: np.ndarray) -> np.ndarray:
        maximum = float(np.max(matrix))
        if not np.isfinite(maximum) or maximum <= 0.0:
            return np.zeros_like(matrix)
        return matrix / maximum

    conductance_score = normalize(conductance)
    tree_score = normalize(tree_marginal)
    diffusion_score = normalize(two_hop)

    score = (
        0.30 * conductance_score
        + 0.45 * tree_score
        + 0.25 * diffusion_score
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.nan_to_num(
        normalize(score),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
