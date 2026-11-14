import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic spectral graph priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distances, np.inf)

    values = distances[valid]
    positive = values[values > 0.0]

    if positive.size:
        center = float(np.median(positive))
        mad = float(np.median(np.abs(positive - center)))
        scale = max(mad, 0.20 * center, eps)
    else:
        center = 1.0
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_max = float(np.max(values))
    if not np.isfinite(finite_max):
        finite_max = center

    replacement = finite_max + 12.0 * scale
    filled = np.where(valid, distances, replacement)
    np.fill_diagonal(filled, replacement)

    transformed = np.exp(
        -np.minimum(filled / scale, 50.0)
    )
    transformed[~valid] = 0.0
    np.fill_diagonal(transformed, 0.0)

    weight_max = float(np.max(transformed))
    if not np.isfinite(weight_max) or weight_max <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    transformed /= weight_max

    degree = transformed.sum(axis=1)
    safe_degree = np.maximum(degree, eps)

    normalized_adjacency = transformed / np.sqrt(
        safe_degree[:, None] * safe_degree[None, :]
    )

    laplacian = np.diag(degree) - transformed
    laplacian = 0.5 * (laplacian + laplacian.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(n, dtype=float)
        eigenvectors = np.eye(n, dtype=float)

    eigenvalues = np.nan_to_num(
        eigenvalues,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    eigenvectors = np.nan_to_num(
        eigenvectors,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    spectral_scale = max(
        float(np.max(np.abs(eigenvalues))),
        eps,
    )
    tolerance = max(
        spectral_scale * 1e-10,
        eps,
    )

    inverse_eigenvalues = np.zeros_like(eigenvalues)
    nonzero = eigenvalues > tolerance
    inverse_eigenvalues[nonzero] = 1.0 / eigenvalues[nonzero]

    laplacian_pinv = (
        eigenvectors
        * inverse_eigenvalues[None, :]
    ) @ eigenvectors.T
    laplacian_pinv = 0.5 * (
        laplacian_pinv + laplacian_pinv.T
    )

    diagonal_pinv = np.diag(laplacian_pinv)
    resistance = (
        diagonal_pinv[:, None]
        + diagonal_pinv[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance[~valid] = 0.0
    np.fill_diagonal(resistance, 0.0)

    resistance_values = resistance[valid]
    resistance_high = (
        float(np.quantile(resistance_values, 0.90))
        if resistance_values.size
        else 1.0
    )
    resistance_low = (
        float(np.quantile(resistance_values, 0.10))
        if resistance_values.size
        else 0.0
    )
    resistance_span = max(
        resistance_high - resistance_low,
        eps,
    )

    resistance_score = np.clip(
        (resistance - resistance_low) / resistance_span,
        0.0,
        1.0,
    )
    resistance_score[~valid] = 0.0

    distance_values = distances[valid]
    distance_order = np.argsort(
        distance_values,
        kind="stable",
    )
    distance_ranks = np.empty(
        distance_values.size,
        dtype=float,
    )
    distance_ranks[distance_order] = np.arange(
        distance_values.size,
        dtype=float,
    )

    distance_score_values = 1.0 - (
        distance_ranks / max(
            float(distance_values.size - 1),
            1.0,
        )
    )

    distance_score = np.zeros((n, n), dtype=float)
    upper_i, upper_j = np.where(
        np.triu(valid, k=1)
    )
    pair_values = distances[upper_i, upper_j]

    pair_order = np.argsort(
        pair_values,
        kind="stable",
    )
    pair_ranks = np.empty(
        pair_values.size,
        dtype=float,
    )
    pair_ranks[pair_order] = np.arange(
        pair_values.size,
        dtype=float,
    )

    pair_score = 1.0 - (
        pair_ranks / max(
            float(pair_values.size - 1),
            1.0,
        )
    )

    distance_score[upper_i, upper_j] = pair_score
    distance_score[upper_j, upper_i] = pair_score

    spectral_coordinates = (
        eigenvectors
        * np.sqrt(
            np.maximum(
                inverse_eigenvalues,
                0.0,
            )
        )[None, :]
    )

    spectral_distance = np.zeros(
        (n, n),
        dtype=float,
    )

    for component in range(n):
        delta = (
            spectral_coordinates[:, component, None]
            - spectral_coordinates[None, :, component]
        )
        spectral_distance += delta * delta

    spectral_distance = np.sqrt(
        np.maximum(spectral_distance, 0.0)
    )
    spectral_distance[~valid] = 0.0

    spectral_values = spectral_distance[valid]
    if spectral_values.size:
        spectral_q90 = float(
            np.quantile(spectral_values, 0.90)
        )
    else:
        spectral_q90 = 1.0

    spectral_score = np.exp(
        -spectral_distance / max(spectral_q90, eps)
    )
    spectral_score[~valid] = 0.0

    accessibility = normalized_adjacency.sum(axis=1)
    accessibility /= max(
        float(np.max(accessibility)),
        eps,
    )

    endpoint_access = np.sqrt(
        np.maximum(
            accessibility[:, None] * accessibility[None, :],
            0.0,
        )
    )
    endpoint_access[~valid] = 0.0

    resistance_weight = 0.35 + 0.65 * resistance_score
    distance_weight = 0.25 + 0.75 * distance_score
    spectral_weight = 0.30 + 0.70 * spectral_score
    access_weight = 0.50 + 0.50 * endpoint_access

    score = (
        distance_weight
        * resistance_weight
        * spectral_weight
        * access_weight
    )

    score[~valid] = 0.0
    score = 0.5 * (
        score + score.T
    )
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    result = np.power(
        np.maximum(score, 0.0),
        0.85,
    )
    result[~valid] = 0.0
    result = 0.5 * (
        result + result.T
    )
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
