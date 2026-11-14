import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally informed symmetric edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    d = 0.5 * (distances + distances.T)
    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    positive = d[valid & (d > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        finite_values = d[valid]
        scale = float(np.max(finite_values)) if finite_values.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-np.clip(d[valid] / scale, 0.0, 60.0))
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)

    degree = affinity.sum(axis=1)
    if not np.any(degree > 0.0):
        return result

    laplacian = np.diag(degree) - affinity

    try:
        laplacian_pinv = np.linalg.pinv(
            laplacian,
            hermitian=True,
            rcond=1e-10,
        )
    except TypeError:
        laplacian_pinv = np.linalg.pinv(laplacian, rcond=1e-10)

    diagonal = np.diag(laplacian_pinv)
    resistance = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance = 0.5 * (resistance + resistance.T)

    resistance_values = resistance[valid & np.isfinite(resistance)]
    if resistance_values.size:
        resistance_scale = float(np.median(resistance_values))
    else:
        resistance_scale = 1.0

    if not np.isfinite(resistance_scale) or resistance_scale <= 0.0:
        resistance_scale = 1.0

    connectivity = np.exp(
        -np.clip(resistance / resistance_scale, 0.0, 60.0)
    )

    score = affinity * connectivity
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
