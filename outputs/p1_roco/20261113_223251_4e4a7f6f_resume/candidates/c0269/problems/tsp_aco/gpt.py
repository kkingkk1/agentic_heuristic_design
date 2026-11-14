import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-diffusion edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0

    valid = finite & nonnegative
    valid &= valid.T
    valid &= ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distances, np.inf)

    edge_values = distances[valid]
    positive = edge_values[edge_values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        scale = max(mad, 0.25 * median)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    normalized = np.zeros((n, n), dtype=float)
    normalized[valid] = distances[valid] / safe_scale
    normalized = np.minimum(normalized, 40.0)

    local_affinity = np.zeros((n, n), dtype=float)
    local_affinity[valid] = np.exp(
        -0.5 * normalized[valid] ** 2
    )

    affinity_mass = local_affinity.sum(axis=1)
    if not np.any(affinity_mass > 0.0):
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        result /= max(float(np.max(result)), 1.0)
        return result

    transition = local_affinity / np.maximum(
        affinity_mass[:, None],
        eps,
    )

    symmetric_affinity = 0.5 * (
        local_affinity + local_affinity.T
    )
    degree = symmetric_affinity.sum(axis=1)

    inv_sqrt_degree = 1.0 / np.sqrt(
        np.maximum(degree, eps)
    )
    normalized_affinity = (
        inv_sqrt_degree[:, None]
        * symmetric_affinity
        * inv_sqrt_degree[None, :]
    )

    laplacian = np.eye(n, dtype=float) - normalized_affinity
    laplacian = 0.5 * (laplacian + laplacian.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(n, dtype=float)
        eigenvectors = np.eye(n, dtype=float)

    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    component_count = min(
        n,
        max(2, int(np.ceil(np.sqrt(float(n)))) + 1),
    )

    usable = min(
        component_count,
        max(1, n - 1),
    )

    embedding = eigenvectors[:, :usable]
    spectral_scale = np.sqrt(
        np.maximum(
            eigenvalues[:usable] - np.min(eigenvalues[:usable]),
            0.0,
        )
        + 0.08
    )

    embedding = embedding / spectral_scale[None, :]

    spectral_distance = np.zeros(
        (n, n),
        dtype=float,
    )

    for component in range(usable):
        difference = (
            embedding[:, component, None]
            - embedding[None, :, component]
        )
        spectral_distance += difference * difference

    spectral_distance = np.sqrt(
        np.maximum(spectral_distance, 0.0)
    )

    finite_spectral = spectral_distance[valid]
    spectral_scale_value = (
        float(np.median(finite_spectral))
        if finite_spectral.size
        else 1.0
    )
    spectral_scale_value = max(
        spectral_scale_value,
        np.sqrt(eps),
    )

    spectral_coherence = np.zeros(
        (n, n),
        dtype=float,
    )
    spectral_coherence[valid] = np.exp(
        -(
            spectral_distance[valid]
            / spectral_scale_value
        ) ** 2
    )

    walk_two = transition @ transition
    walk_two = 0.5 * (walk_two + walk_two.T)

    walk_mass = np.maximum(
        np.max(walk_two[valid]) if np.any(valid) else 0.0,
        eps,
    )
    walk_support = np.zeros(
        (n, n),
        dtype=float,
    )
    walk_support[valid] = np.clip(
        walk_two[valid] / walk_mass,
        0.0,
        1.0,
    )

    combined = (
        np.maximum(local_affinity, eps) ** 0.52
        * np.maximum(spectral_coherence, eps) ** 0.30
        * np.maximum(walk_support, eps) ** 0.18
    )
    combined[~valid] = 0.0
    combined = 0.5 * (
        combined + combined.T
    )
    combined[~valid] = 0.0
    np.fill_diagonal(combined, 0.0)

    row_mass = combined.sum(axis=1)
    isolated = row_mass <= eps

    if np.any(isolated):
        fallback = np.zeros_like(combined)
        fallback[valid] = local_affinity[valid]
        fallback = 0.5 * (
            fallback + fallback.T
        )
        combined[isolated] = fallback[isolated]
        combined[:, isolated] = fallback[:, isolated]
        combined[~valid] = 0.0
        np.fill_diagonal(combined, 0.0)

    scale_factors = np.ones(n, dtype=float)

    for _ in range(32):
        row_mass = combined.sum(axis=1)
        positive_rows = row_mass > eps

        update = np.ones(n, dtype=float)
        update[positive_rows] = 1.0 / np.sqrt(
            row_mass[positive_rows]
        )

        scale_factors *= update
        scale_factors = np.clip(
            scale_factors,
            1e-8,
            1e8,
        )

        scaled = (
            scale_factors[:, None]
            * combined
            * scale_factors[None, :]
        )
        scaled[~valid] = 0.0
        np.fill_diagonal(scaled, 0.0)

        new_mass = scaled.sum(axis=1)
        correction = np.ones(n, dtype=float)
        active = new_mass > eps
        correction[active] = 1.0 / np.sqrt(
            new_mass[active]
        )

        scale_factors *= correction
        scale_factors = np.clip(
            scale_factors,
            1e-8,
            1e8,
        )

    result = (
        scale_factors[:, None]
        * combined
        * scale_factors[None, :]
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
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
