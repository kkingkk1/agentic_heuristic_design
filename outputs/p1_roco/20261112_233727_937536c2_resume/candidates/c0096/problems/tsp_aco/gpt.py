import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d)
    valid = finite & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    values = d[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.median(values)) if values.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    safe = np.full((n, n), 4.0 * scale, dtype=float)
    safe[valid] = d[valid]
    safe = np.minimum(safe, safe.T)
    np.fill_diagonal(safe, 0.0)

    usable = np.ones((n, n), dtype=bool)
    np.fill_diagonal(usable, False)

    normalized = safe / scale
    affinity = np.exp(-0.5 * np.square(np.clip(normalized, 0.0, 50.0)))
    affinity[~usable] = 0.0

    degree = affinity.sum(axis=1)
    degree = np.maximum(degree, eps)

    inv_sqrt_degree = 1.0 / np.sqrt(degree)
    normalized_affinity = (
        inv_sqrt_degree[:, None]
        * affinity
        * inv_sqrt_degree[None, :]
    )

    laplacian = np.eye(n, dtype=float) - normalized_affinity
    laplacian = 0.5 * (laplacian + laplacian.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        order = np.argsort(eigenvalues, kind="stable")
        eigenvectors = eigenvectors[:, order]
    except np.linalg.LinAlgError:
        eigenvectors = np.zeros((n, n), dtype=float)

    if n >= 3:
        x = eigenvectors[:, 1]
        y = eigenvectors[:, 2]
    else:
        x = np.arange(n, dtype=float)
        y = np.zeros(n, dtype=float)

    radius = np.sqrt(x * x + y * y)
    radius_scale = np.median(radius[radius > eps]) if np.any(radius > eps) else 1.0
    radius_scale = max(float(radius_scale), eps)

    angles = np.arctan2(y, x)
    angle_difference = np.abs(angles[:, None] - angles[None, :])
    angle_difference = np.minimum(
        angle_difference,
        2.0 * np.pi - angle_difference,
    )

    angular_affinity = np.exp(
        -angle_difference / (np.pi / 3.0)
    )

    radial_difference = np.abs(
        radius[:, None] - radius[None, :]
    ) / radius_scale
    radial_affinity = np.exp(
        -0.5 * np.square(np.clip(radial_difference, 0.0, 20.0))
    )

    distance_affinity = np.exp(
        -0.15 * np.square(np.clip(normalized, 0.0, 20.0))
    )

    score = (
        angular_affinity
        * (0.65 + 0.35 * radial_affinity)
        * (0.80 + 0.20 * distance_affinity)
    )

    score[~valid & ~valid.T] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float)
