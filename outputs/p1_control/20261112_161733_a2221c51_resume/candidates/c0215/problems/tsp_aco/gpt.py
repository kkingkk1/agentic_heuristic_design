import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-structural edge indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d) & (d >= 0.0)
    d = np.where(finite, d, 0.0)
    d = 0.5 * (d + d.T)

    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    scale = max(scale, eps)

    finite_valid_distances = d[valid]
    largest = float(np.max(finite_valid_distances))
    replacement = max(largest * 4.0, scale * 4.0, eps)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    # Robust conductance kernel. Shorter edges receive stronger conductance.
    conductance = np.exp(-np.minimum(work / scale, 60.0))
    conductance[~valid] = 0.0
    np.fill_diagonal(conductance, 0.0)

    # Mildly suppress edges whose conductance is negligible everywhere.
    row_mass = np.sum(conductance, axis=1)
    row_mass = np.maximum(row_mass, eps)
    local_support = conductance / np.sqrt(
        row_mass[:, None] * row_mass[None, :]
    )
    local_support = np.maximum(local_support, 0.0)

    # Effective resistance captures global graph-level edge importance.
    degree = np.sum(conductance, axis=1)
    laplacian = np.diag(degree) - conductance

    # Numerical symmetrization improves stability before eigendecomposition.
    laplacian = 0.5 * (laplacian + laplacian.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        cutoff = max(float(np.max(eigenvalues)) * 1e-10, eps)
        inverse_values = np.where(
            eigenvalues > cutoff,
            1.0 / eigenvalues,
            0.0,
        )
        laplacian_pseudoinverse = (
            eigenvectors * inverse_values[None, :]
        ) @ eigenvectors.T
        laplacian_pseudoinverse = 0.5 * (
            laplacian_pseudoinverse
            + laplacian_pseudoinverse.T
        )

        diagonal = np.diag(laplacian_pseudoinverse)
        resistance = (
            diagonal[:, None]
            + diagonal[None, :]
            - 2.0 * laplacian_pseudoinverse
        )
        resistance = np.maximum(resistance, 0.0)
    except np.linalg.LinAlgError:
        resistance = np.zeros((n, n), dtype=float)

    resistance[~valid] = 0.0
    np.fill_diagonal(resistance, 0.0)

    leverage = conductance * resistance
    leverage = np.maximum(leverage, 0.0)

    leverage_values = leverage[valid]
    if leverage_values.size:
        leverage_scale = float(np.median(leverage_values))
    else:
        leverage_scale = 1.0

    leverage_scale = max(leverage_scale, eps)
    normalized_leverage = np.minimum(leverage / leverage_scale, 20.0)

    # Combine local support and global structural importance.
    score = (
        np.power(np.maximum(local_support, eps), 0.75)
        * (0.55 + 0.45 * np.exp(-0.35 * normalized_leverage))
        * (1.0 + 0.80 * np.tanh(normalized_leverage))
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
