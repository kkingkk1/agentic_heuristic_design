import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = d[finite_positive]
    fallback = float(np.median(positive_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.array(d, dtype=float, copy=True)
    clean[~np.isfinite(clean)] = fallback
    clean[clean <= 0] = fallback

    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    observed = clean[off_diagonal]
    scale = float(np.median(observed))
    if not np.isfinite(scale) or scale <= 0:
        scale = fallback

    scale = max(scale, np.finfo(float).eps)

    affinity = np.exp(-np.square(clean / scale))
    np.fill_diagonal(affinity, 0.0)

    degree = np.sum(affinity, axis=1)
    degree = np.maximum(degree, np.finfo(float).eps)

    inv_sqrt_degree = 1.0 / np.sqrt(degree)
    normalized_affinity = (
        inv_sqrt_degree[:, None]
        * affinity
        * inv_sqrt_degree[None, :]
    )

    normalized_affinity = 0.5 * (
        normalized_affinity + normalized_affinity.T
    )

    laplacian = np.eye(n, dtype=float) - normalized_affinity
    laplacian = 0.5 * (laplacian + laplacian.T)

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    diffusion_time = np.log1p(n) / max(1.0, np.sqrt(n))
    heat_weights = np.exp(-diffusion_time * np.maximum(eigenvalues, 0.0))

    heat_kernel = (
        eigenvectors
        @ (heat_weights[:, None] * eigenvectors.T)
    )

    heat_kernel = np.maximum(heat_kernel, 0.0)
    heat_kernel = 0.5 * (heat_kernel + heat_kernel.T)
    np.fill_diagonal(heat_kernel, 0.0)

    heat_kernel *= off_diagonal
    maximum = float(np.max(heat_kernel))

    if maximum <= 0 or not np.isfinite(maximum):
        return np.zeros((n, n), dtype=float)

    scores = heat_kernel / maximum
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
