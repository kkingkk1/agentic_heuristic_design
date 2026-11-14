import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric graph-structural prior for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.zeros_like(d, dtype=float)
    symmetric[valid] = 0.5 * (d[valid] + d.T[valid])

    values = symmetric[valid]
    eps = np.finfo(float).eps
    scale = max(float(np.median(values)), eps)

    x = np.zeros_like(symmetric, dtype=float)
    x[valid] = np.minimum(symmetric[valid] / scale, 1e12)

    if n == 2:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    positive_x = x[valid]
    temperature = max(float(np.median(positive_x)), 1e-6)

    best_two_hop = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        through_k = x[:, [k]] + x[[k], :]
        usable = valid[:, [k]] & valid[[k], :]
        usable[:, k] = False
        usable[k, :] = False
        through_k[~usable] = np.inf
        best_two_hop = np.minimum(best_two_hop, through_k)

    direct = np.maximum(x, eps)
    finite_detours = np.isfinite(best_two_hop) & valid

    detour_gain = np.zeros((n, n), dtype=float)
    detour_gain[finite_detours] = (
        best_two_hop[finite_detours] - direct[finite_detours]
    )
    detour_gain = np.maximum(detour_gain, 0.0)
    detour_gain = np.minimum(detour_gain, 50.0)

    detour_scale = np.where(
        finite_detours,
        best_two_hop + direct + eps,
        1.0,
    )
    structural_necessity = detour_gain / detour_scale

    affinity = np.exp(-np.minimum(direct / temperature, 50.0))
    affinity[~valid] = 0.0

    incident_mass = np.sum(
        np.where(
            valid,
            np.exp(-np.minimum(x / temperature, 50.0)),
            0.0,
        ),
        axis=1,
    )
    endpoint_balance = 1.0 / np.sqrt(
        (1.0 + incident_mass[:, None])
        * (1.0 + incident_mass[None, :])
    )

    endpoint_strength = np.sqrt(
        np.maximum(affinity, eps)
        * np.maximum(affinity.T, eps)
    )

    log_score = (
        3.0 * structural_necessity
        + 0.8 * np.log(np.maximum(endpoint_strength, eps))
        + 0.35 * np.log(np.maximum(endpoint_balance, eps))
    )

    log_score[~valid] = -np.inf
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
