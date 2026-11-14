import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid_forward = np.isfinite(d) & (d > 0)
    valid_reverse = valid_forward.T
    valid = valid_forward | valid_reverse

    forward = np.where(valid_forward, d, np.inf)
    reverse = np.where(valid_reverse, d.T, np.inf)

    both_valid = valid_forward & valid_reverse
    effective_distance = np.where(
        both_valid,
        0.5 * (forward + reverse),
        np.minimum(forward, reverse),
    )
    np.fill_diagonal(effective_distance, np.inf)
    valid &= ~np.eye(n, dtype=bool)

    values = effective_distance[np.isfinite(effective_distance)]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = np.median(values)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(values)
    scale = max(float(scale), np.finfo(float).eps)

    proximity = np.exp(-effective_distance / scale)
    proximity[~np.isfinite(proximity)] = 0.0
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    row_mass = proximity.sum(axis=1)
    safe_mass = np.maximum(row_mass, np.finfo(float).eps)

    scarcity = 1.0 / safe_mass
    scarcity /= max(float(scarcity.mean()), np.finfo(float).eps)
    endpoint_factor = np.sqrt(scarcity[:, None] * scarcity[None, :])

    local_support = proximity / safe_mass[:, None]
    mutual_support = np.sqrt(
        np.maximum(local_support * local_support.T, 0.0)
    )

    inverse_distance = scale / np.maximum(effective_distance, scale)
    inverse_distance[~valid] = 0.0

    score = (
        inverse_distance
        * (0.60 + 0.35 * endpoint_factor)
        * (0.75 + 0.75 * mutual_support)
        * (0.80 + 0.20 * proximity)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = score.max(initial=0.0)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
