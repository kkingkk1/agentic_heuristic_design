import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric graph-structural prior for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite_nonnegative = np.isfinite(d) & (d >= 0)
    valid = finite_nonnegative & finite_nonnegative.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    symmetric_distance = 0.5 * (d + d.T)
    positive = symmetric_distance[valid & (symmetric_distance > 0)]

    if positive.size:
        scale = float(np.median(positive))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.mean(positive))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    finite_values = symmetric_distance[valid]
    spread = float(np.median(np.abs(finite_values - np.median(finite_values))))
    temperature = max(scale + spread, np.finfo(float).eps)

    affinity = np.zeros((n, n), dtype=float)
    normalized_distance = symmetric_distance / temperature
    affinity[valid] = np.exp(-np.clip(normalized_distance[valid], 0.0, 60.0))

    active = affinity.sum(axis=1) > 0
    if not np.all(active):
        valid &= active[:, None] & active[None, :]
        affinity[~valid] = 0.0

    if not np.any(valid):
        return result

    scaling = np.ones(n, dtype=float)

    for _ in range(100):
        row_mass = affinity @ scaling
        updated = np.zeros_like(scaling)
        positive_mass = row_mass > np.finfo(float).eps
        updated[positive_mass] = 1.0 / row_mass[positive_mass]
        updated[~positive_mass] = 0.0

        if np.max(np.abs(updated - scaling)) < 1e-10:
            scaling = updated
            break

        scaling = 0.5 * scaling + 0.5 * updated

    balanced = scaling[:, None] * affinity * scaling[None, :]
    balanced[~valid] = 0.0
    balanced = 0.5 * (balanced + balanced.T)

    degree = balanced.sum(axis=1)
    laplacian = np.diag(degree) - balanced

    try:
        laplacian_pinv = np.linalg.pinv(laplacian, rcond=1e-10)
    except np.linalg.LinAlgError:
        return balanced

    diagonal = np.diag(laplacian_pinv)
    effective_resistance = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2.0 * laplacian_pinv
    )
    effective_resistance = np.maximum(effective_resistance, 0.0)
    effective_resistance[~valid] = 0.0
    np.fill_diagonal(effective_resistance, 0.0)

    resistance_values = effective_resistance[effective_resistance > 0]
    resistance_scale = (
        float(np.median(resistance_values))
        if resistance_values.size
        else 1.0
    )
    resistance_scale = max(resistance_scale, np.finfo(float).eps)

    score = balanced * (
        0.35 + 0.65 * effective_resistance / resistance_scale
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
