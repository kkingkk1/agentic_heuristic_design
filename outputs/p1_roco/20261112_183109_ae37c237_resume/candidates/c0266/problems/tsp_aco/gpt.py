import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric spectral-structural edge priorities for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return result

    with np.errstate(all="ignore"):
        d = 0.5 * (distances + distances.T)

    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    values = d[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    elif values.size:
        scale = float(np.max(values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    normalized = np.divide(
        d,
        scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = 1.0 / (
        1.0 + np.square(np.minimum(normalized[valid], 1.0e6))
    )
    np.fill_diagonal(affinity, 0.0)

    degree = affinity.sum(axis=1)
    inv_sqrt_degree = np.zeros(n, dtype=float)
    positive_degree = degree > 0.0
    inv_sqrt_degree[positive_degree] = 1.0 / np.sqrt(
        degree[positive_degree]
    )

    normalized_affinity = (
        inv_sqrt_degree[:, None]
        * affinity
        * inv_sqrt_degree[None, :]
    )

    laplacian = np.eye(n, dtype=float) - normalized_affinity
    laplacian = 0.5 * (laplacian + laplacian.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(n, dtype=float)
        eigenvectors = np.eye(n, dtype=float)

    del eigenvalues

    spectral_parts = []
    if n >= 3:
        spectral_parts.append(eigenvectors[:, 1])
    if n >= 4:
        spectral_parts.append(eigenvectors[:, 2])

    if spectral_parts:
        embedding = np.column_stack(spectral_parts)
        embedding = embedding * inv_sqrt_degree[:, None]

        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        unit_embedding = np.divide(
            embedding,
            norms,
            out=np.zeros_like(embedding),
            where=norms > eps,
        )

        spectral_similarity = unit_embedding @ unit_embedding.T
        spectral_similarity = np.clip(
            0.5 * (spectral_similarity + 1.0),
            0.0,
            1.0,
        )
    else:
        spectral_similarity = np.zeros((n, n), dtype=float)

    common_support = affinity @ affinity.T
    support_norm = np.sqrt(
        np.maximum(
            np.diag(common_support)[:, None]
            * np.diag(common_support)[None, :],
            eps,
        )
    )
    common_support = np.divide(
        common_support,
        support_norm,
        out=np.zeros_like(common_support),
        where=support_norm > 0.0,
    )
    common_support = np.clip(common_support, 0.0, 1.0)

    triangle_cost = np.full((n, n), np.inf, dtype=float)
    for pivot in range(n):
        via_pivot = d[:, pivot, None] + d[pivot, None, :]
        triangle_cost = np.minimum(triangle_cost, via_pivot)

    detour_ratio = np.divide(
        triangle_cost,
        d,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (d > 0.0),
    )
    detour_quality = np.exp(
        -np.clip(np.maximum(detour_ratio - 1.0, 0.0), 0.0, 40.0)
    )
    detour_quality[~valid] = 0.0

    distance_similarity = np.exp(
        -np.clip(normalized, 0.0, 40.0)
    )
    distance_similarity[~valid] = 0.0

    scores = (
        distance_similarity
        * (
            0.34
            + 0.30 * spectral_similarity
            + 0.20 * common_support
            + 0.16 * detour_quality
        )
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
