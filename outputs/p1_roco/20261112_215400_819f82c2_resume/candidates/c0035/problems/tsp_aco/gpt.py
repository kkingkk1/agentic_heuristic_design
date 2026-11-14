import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric effective-resistance edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    valid_values = distances[finite]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.max(valid_values)) if valid_values.size else 1.0

    scale = max(scale, np.finfo(float).eps)

    symmetric_valid = finite | finite.T
    symmetric_distance = np.full((n, n), np.inf, dtype=float)

    both_valid = finite & finite.T
    symmetric_distance[both_valid] = np.minimum(
        distances[both_valid],
        distances.T[both_valid],
    )

    only_forward = finite & ~finite.T
    symmetric_distance[only_forward] = distances[only_forward]

    only_backward = finite.T & ~finite
    symmetric_distance[only_backward] = distances.T[only_backward]

    affinity = np.zeros((n, n), dtype=float)
    affinity[symmetric_valid] = np.exp(
        -np.clip(symmetric_distance[symmetric_valid] / scale, 0.0, 60.0)
    )
    np.fill_diagonal(affinity, 0.0)
    affinity = 0.5 * (affinity + affinity.T)

    if not np.any(affinity > 0.0):
        return np.zeros((n, n), dtype=float)

    balancing = np.ones(n, dtype=float)
    for _ in range(80):
        row_mass = affinity @ balancing
        updated = np.divide(
            1.0,
            row_mass,
            out=np.zeros_like(row_mass),
            where=row_mass > 0.0,
        )
        updated = np.sqrt(np.maximum(updated, np.finfo(float).eps))
        if np.max(np.abs(updated - balancing)) < 1e-10:
            balancing = updated
            break
        balancing = updated

    balanced = balancing[:, None] * affinity * balancing[None, :]
    balanced = 0.5 * (balanced + balanced.T)
    np.fill_diagonal(balanced, 0.0)

    degree = balanced.sum(axis=1)
    laplacian = np.diag(degree) - balanced

    try:
        laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    except TypeError:
        laplacian_pinv = np.linalg.pinv(laplacian)

    resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance[~symmetric_valid] = 0.0
    np.fill_diagonal(resistance, 0.0)

    positive_resistance = resistance[(balanced > 0.0) & (resistance > 0.0)]
    if positive_resistance.size:
        resistance_scale = float(np.median(positive_resistance))
    else:
        resistance_scale = 1.0
    resistance_scale = max(resistance_scale, np.finfo(float).eps)

    redundancy = np.exp(
        -np.clip(resistance / resistance_scale, 0.0, 60.0)
    )
    score = balanced * (0.25 + 0.75 * redundancy)

    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
