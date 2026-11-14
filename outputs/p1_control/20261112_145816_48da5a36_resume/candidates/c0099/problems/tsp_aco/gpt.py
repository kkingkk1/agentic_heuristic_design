import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_positive = np.isfinite(d) & (d > 0)
    finite_positive &= ~np.eye(n, dtype=bool)

    if not np.any(finite_positive):
        return result

    d_sym = 0.5 * (d + d.T)
    valid = finite_positive & finite_positive.T

    positive_values = d_sym[valid]
    if positive_values.size == 0:
        return result

    scale = float(np.median(positive_values))
    scale = max(scale, np.finfo(float).eps)

    fallback = max(float(np.max(positive_values)) * 10.0, scale)
    safe = np.where(valid, d_sym, fallback)
    np.fill_diagonal(safe, fallback)

    best_two_hop = np.full((n, n), np.inf, dtype=float)
    alternative_count = np.zeros((n, n), dtype=float)

    tolerance = 0.35 * scale

    for k in range(n):
        usable = valid[:, k][:, None] & valid[k, :][None, :]
        two_hop = safe[:, k][:, None] + safe[k, :][None, :]
        best_two_hop = np.minimum(
            best_two_hop,
            np.where(usable, two_hop, np.inf),
        )

    for k in range(n):
        usable = valid[:, k][:, None] & valid[k, :][None, :]
        two_hop = safe[:, k][:, None] + safe[k, :][None, :]
        competitive = usable & (two_hop <= safe + tolerance)
        alternative_count += competitive.astype(float)

    finite_detour = np.isfinite(best_two_hop)
    detour_gap = np.where(
        finite_detour,
        best_two_hop - safe,
        0.0,
    )

    relative_detour = np.tanh(detour_gap / (scale + np.finfo(float).eps))
    scarcity = 1.0 / (1.0 + alternative_count)

    global_length = safe / scale
    log_score = (
        1.55 * relative_detour
        + 0.95 * scarcity
        - 0.22 * np.minimum(global_length, 20.0)
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return result

    maximum = float(np.max(log_score[finite_scores]))
    result = np.exp(np.where(finite_scores, log_score - maximum, -np.inf))
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum

    return result.astype(float, copy=False)
