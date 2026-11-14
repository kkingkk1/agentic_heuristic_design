import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic graph-theoretic edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

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

    values = distances[valid]
    positive = values[values > 0.0]

    if positive.size:
        q25, q50, q75 = np.quantile(
            positive,
            [0.25, 0.50, 0.75],
        )
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(q50)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    finite_max = float(np.max(values))
    temperature = max(
        0.55 * safe_scale,
        0.18 * float(np.median(values)),
        eps,
    )

    weights = np.zeros((n, n), dtype=float)
    scaled = np.minimum(distances / temperature, 60.0)
    weights[valid] = np.exp(-scaled[valid])
    weights = 0.5 * (weights + weights.T)
    np.fill_diagonal(weights, 0.0)

    degree = weights.sum(axis=1)
    active = degree > eps

    if not np.any(active):
        return np.zeros((n, n), dtype=float)

    laplacian = np.diag(degree) - weights

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    spectral_scale = max(
        float(np.max(eigenvalues)),
        1.0,
    )
    tolerance = eps * max(n, 1) * spectral_scale

    inverse_eigenvalues = np.zeros_like(eigenvalues)
    positive_eigenvalues = eigenvalues > tolerance
    inverse_eigenvalues[positive_eigenvalues] = (
        1.0 / eigenvalues[positive_eigenvalues]
    )

    laplacian_inverse = (
        eigenvectors
        @ np.diag(inverse_eigenvalues)
        @ eigenvectors.T
    )
    laplacian_inverse = 0.5 * (
        laplacian_inverse + laplacian_inverse.T
    )

    diagonal_inverse = np.diag(laplacian_inverse)
    effective_resistance = (
        diagonal_inverse[:, None]
        + diagonal_inverse[None, :]
        - 2.0 * laplacian_inverse
    )
    effective_resistance = np.maximum(
        effective_resistance,
        0.0,
    )
    effective_resistance[~valid] = 0.0
    np.fill_diagonal(effective_resistance, 0.0)

    resistance_values = effective_resistance[valid]
    resistance_max = (
        float(np.quantile(resistance_values, 0.90))
        if resistance_values.size
        else 1.0
    )
    resistance_max = max(resistance_max, eps)

    leverage = np.zeros((n, n), dtype=float)
    leverage[valid] = np.clip(
        effective_resistance[valid] / resistance_max,
        0.0,
        1.0,
    )

    normalized_degree = degree / max(
        float(np.max(degree)),
        eps,
    )
    endpoint_balance = np.sqrt(
        np.maximum(
            normalized_degree[:, None]
            * normalized_degree[None, :],
            0.0,
        )
    )

    common_support = weights @ weights
    common_support[~valid] = 0.0
    np.fill_diagonal(common_support, 0.0)

    support_values = common_support[valid]
    support_scale = (
        float(np.quantile(support_values, 0.75))
        if support_values.size
        else 1.0
    )
    support_scale = max(support_scale, eps)

    closure = np.zeros((n, n), dtype=float)
    closure[valid] = np.clip(
        common_support[valid] / support_scale,
        0.0,
        1.0,
    )

    distance_component = np.zeros((n, n), dtype=float)
    distance_component[valid] = np.exp(
        -np.minimum(
            distances[valid] / safe_scale,
            60.0,
        )
    )

    resistance_component = 0.30 + 0.70 * leverage
    closure_component = 0.42 + 0.58 * closure
    degree_component = 0.55 + 0.45 * endpoint_balance

    score = (
        np.maximum(distance_component, eps) ** 1.35
        * np.maximum(resistance_component, eps) ** 0.85
        * np.maximum(closure_component, eps) ** 0.75
        * np.maximum(degree_component, eps) ** 0.60
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
