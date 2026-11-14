import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic spectral-diffusion edge priors for stochastic TSP."""
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
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distances, np.inf)

    observed = distances[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        log_values = np.log1p(positive)
        center = float(np.median(log_values))
        spread = float(np.quantile(log_values, 0.75) -
                       np.quantile(log_values, 0.25))
    else:
        center = 0.0
        spread = 1.0

    if not np.isfinite(spread) or spread <= 1e-12:
        spread = max(center, 1.0)

    eps = np.finfo(float).eps
    safe_spread = max(spread, eps)

    transformed = np.zeros((n, n), dtype=float)
    transformed[valid] = np.log1p(distances[valid])

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.abs(transformed[valid] - center) / safe_spread
    )

    # Restore the preference for shorter edges after robust scale compression.
    shortest = np.log1p(np.maximum(distances, 0.0))
    affinity[valid] *= np.exp(
        -shortest[valid] / max(0.85 * safe_spread, eps)
    )

    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    maximum = float(np.max(affinity))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    affinity /= maximum

    degree = affinity.sum(axis=1)
    safe_degree = np.maximum(degree, eps)
    normalized = affinity / np.sqrt(
        safe_degree[:, None] * safe_degree[None, :]
    )
    normalized[~valid] = 0.0
    np.fill_diagonal(normalized, 0.0)

    laplacian = np.eye(n, dtype=float) - normalized
    laplacian = 0.5 * (laplacian + laplacian.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        diffusion_time = 2.4
        heat = (
            eigenvectors
            * np.exp(-diffusion_time * eigenvalues)[None, :]
        ) @ eigenvectors.T
        heat = 0.5 * (heat + heat.T)
        heat = np.maximum(heat, 0.0)
    except np.linalg.LinAlgError:
        heat = normalized.copy()

    heat[~valid] = 0.0
    np.fill_diagonal(heat, 0.0)

    heat_max = float(np.max(heat))
    if np.isfinite(heat_max) and heat_max > 0.0:
        heat /= heat_max
    else:
        heat.fill(0.0)

    score = (
        0.58 * affinity
        + 0.42 * heat
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    score_max = float(np.max(score))
    if np.isfinite(score_max) and score_max > 0.0:
        score /= score_max

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
