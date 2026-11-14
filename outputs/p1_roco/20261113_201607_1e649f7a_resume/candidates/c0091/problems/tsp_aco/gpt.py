import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return transport-balanced structural priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    valid_values = distances[valid]
    positive = valid_values[valid_values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.max(valid_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    largest = float(np.max(valid_values))
    penalty = largest + 8.0 * scale

    clean = np.where(valid, distances, penalty).astype(float, copy=False)

    normalized = np.minimum(clean / scale, 60.0)
    kernel = np.exp(-normalized)
    kernel[~valid] = 0.0

    eps = np.finfo(float).eps
    balanced = kernel.copy()

    for _ in range(48):
        row_mass = balanced.sum(axis=1, keepdims=True)
        row_scale = np.divide(
            1.0,
            row_mass,
            out=np.zeros_like(row_mass),
            where=row_mass > eps,
        )
        balanced *= row_scale

        col_mass = balanced.sum(axis=0, keepdims=True)
        col_scale = np.divide(
            1.0,
            col_mass,
            out=np.zeros_like(col_mass),
            where=col_mass > eps,
        )
        balanced *= col_scale

    bridge = np.full((n, n), np.inf, dtype=float)

    for middle in range(n):
        through_middle = (
            clean[:, middle, None] + clean[None, middle, :]
        )
        supported = valid[:, middle, None] & valid[None, middle, :]
        bridge = np.minimum(
            bridge,
            np.where(supported, through_middle, np.inf),
        )

    detour_gap = np.zeros((n, n), dtype=float)
    replaceable = valid & np.isfinite(bridge)

    detour_gap[replaceable] = np.maximum(
        bridge[replaceable] - distances[replaceable],
        0.0,
    ) / scale

    structural_bonus = np.zeros((n, n), dtype=float)
    structural_bonus[valid] = (
        detour_gap[valid] / (1.0 + detour_gap[valid])
    )

    support = balanced * (0.72 + 0.28 * structural_bonus)
    support[~valid] = 0.0

    maximum = float(np.max(support))
    if np.isfinite(maximum) and maximum > 0.0:
        support /= maximum

    return support.astype(float, copy=False)
