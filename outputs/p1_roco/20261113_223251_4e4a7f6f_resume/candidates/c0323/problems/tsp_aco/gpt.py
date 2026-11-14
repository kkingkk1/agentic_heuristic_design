import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global spectral-diffusion edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = (
        finite_nonnegative
        & finite_nonnegative.T
        & ~diagonal
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)

    observed = distances[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    # Bounded conductance graph. Smaller distances produce larger conductance.
    conductance = np.zeros((n, n), dtype=float)
    conductance[valid] = 1.0 / (
        1.0 + distances[valid] / safe_scale
    )
    conductance = 0.5 * (conductance + conductance.T)
    conductance[~valid] = 0.0
    np.fill_diagonal(conductance, 0.0)

    degree = conductance.sum(axis=1)
    active = degree > eps

    if not np.any(active):
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    inv_sqrt_degree = np.zeros(n, dtype=float)
    inv_sqrt_degree[active] = 1.0 / np.sqrt(
        np.maximum(degree[active], eps)
    )

    normalized_adjacency = (
        inv_sqrt_degree[:, None]
        * conductance
        * inv_sqrt_degree[None, :]
    )
    normalized_adjacency = 0.5 * (
        normalized_adjacency + normalized_adjacency.T
    )

    laplacian = np.eye(n, dtype=float) - normalized_adjacency
    laplacian[~active, :] = 0.0
    laplacian[:, ~active] = 0.0
    laplacian[~active, ~active] = 0.0
    laplacian = 0.5 * (laplacian + laplacian.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(n, dtype=float)
        eigenvectors = np.eye(n, dtype=float)

    eigenvalues = np.nan_to_num(
        eigenvalues,
        nan=1.0,
        posinf=2.0,
        neginf=0.0,
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)

    finite_eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    if finite_eigenvalues.size:
        spectral_scale = float(
            np.median(finite_eigenvalues)
        )
    else:
        spectral_scale = 1.0

    if not np.isfinite(spectral_scale) or spectral_scale <= eps:
        spectral_scale = 1.0

    diffusion_time = 1.0 / spectral_scale
    diffusion_time = float(
        np.clip(diffusion_time, 0.35, 8.0)
    )

    heat_weights = np.exp(
        -diffusion_time * np.minimum(eigenvalues, 60.0)
    )
    heat_kernel = (
        eigenvectors
        @ (heat_weights[:, None] * eigenvectors.T)
    )
    heat_kernel = 0.5 * (
        heat_kernel + heat_kernel.T
    )
    heat_kernel = np.nan_to_num(
        heat_kernel,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    spectral_signal = np.maximum(heat_kernel, 0.0)
    spectral_signal[~valid] = 0.0
    np.fill_diagonal(spectral_signal, 0.0)

    spectral_max = float(np.max(spectral_signal))
    if np.isfinite(spectral_max) and spectral_max > eps:
        spectral_signal /= spectral_max
    else:
        spectral_signal = conductance.copy()
        spectral_signal[~valid] = 0.0

    # Global distance percentile preference, independent of local neighborhoods.
    ordered = np.sort(observed)
    ranks = np.searchsorted(
        ordered,
        observed,
        side="right",
    ).astype(float)

    percentile_preference = np.zeros((n, n), dtype=float)
    percentile_preference[valid] = 1.0 - (
        ranks / max(float(ordered.size), 1.0)
    )
    percentile_preference[valid] = np.maximum(
        percentile_preference[valid],
        0.05,
    )

    # Degree correction rewards edges joining structurally underrepresented nodes.
    degree_mass = np.sqrt(
        np.maximum(degree[:, None] * degree[None, :], eps)
    )
    degree_correction = np.zeros((n, n), dtype=float)
    degree_correction[valid] = 1.0 / degree_mass[valid]

    correction_max = float(np.max(degree_correction))
    if np.isfinite(correction_max) and correction_max > eps:
        degree_correction /= correction_max

    score = (
        np.maximum(spectral_signal, eps) ** 1.55
        * np.maximum(percentile_preference, eps) ** 0.70
        * np.maximum(
            0.35 + 0.65 * degree_correction,
            eps,
        )
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= eps:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    # Preserve a small graph-wide exploration floor without matrix balancing.
    result = 0.92 * score + 0.08 * conductance
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > eps:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
