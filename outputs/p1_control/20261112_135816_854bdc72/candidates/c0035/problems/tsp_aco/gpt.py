import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    # Use a symmetric graph representation.
    symmetric_d = 0.5 * (d + d.T)
    valid = np.isfinite(symmetric_d) & (symmetric_d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive = symmetric_d[valid]
    distance_scale = float(np.median(positive))
    distance_scale = max(distance_scale, np.finfo(float).eps)

    # Conductance interpretation: short edges are strong graph connections.
    conductance = np.zeros((n, n), dtype=float)
    conductance[valid] = distance_scale / symmetric_d[valid]
    conductance = 0.5 * (conductance + conductance.T)
    np.fill_diagonal(conductance, 0.0)

    # Weighted graph Laplacian.
    laplacian = np.diag(np.sum(conductance, axis=1)) - conductance

    # Effective resistance from the Laplacian pseudoinverse.
    # The eigen formulation is numerically safer than explicitly inverting L.
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    tolerance = max(
        np.finfo(float).eps * max(1.0, float(np.max(np.abs(eigenvalues)))) * n,
        1e-12,
    )

    positive_modes = eigenvalues > tolerance
    if np.any(positive_modes):
        projected = eigenvectors[:, positive_modes]
        inverse_values = 1.0 / eigenvalues[positive_modes]
        laplacian_pinv = (
            projected * inverse_values[None, :]
        ) @ projected.T
    else:
        laplacian_pinv = np.zeros((n, n), dtype=float)

    effective_resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    effective_resistance = np.maximum(effective_resistance, 0.0)

    # Weighted spanning-tree leverage: w_ij * R_ij.
    leverage = conductance * effective_resistance

    # Retain a mild preference for shorter edges without local statistics.
    max_conductance = float(np.max(conductance))
    if max_conductance > 0:
        normalized_conductance = conductance / max_conductance
    else:
        normalized_conductance = conductance

    score = leverage * np.power(np.maximum(normalized_conductance, 0.0), 0.25)
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
