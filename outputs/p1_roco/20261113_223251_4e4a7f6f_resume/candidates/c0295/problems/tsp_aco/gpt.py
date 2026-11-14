import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal_mask = np.eye(n, dtype=bool)
    finite_offdiag = np.isfinite(raw) & ~diagonal_mask

    if not np.any(finite_offdiag):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(np.isfinite(raw), raw, 0.0)
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[finite_offdiag]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_max = float(np.max(observed)) if observed.size else scale
    penalty = finite_max + 16.0 * scale

    distances = np.where(finite_offdiag, cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    valid_distances = distances[~diagonal_mask]
    valid_distances = valid_distances[valid_distances > 0.0]
    temperature = (
        float(np.median(valid_distances)) if valid_distances.size else scale
    )
    temperature = max(0.20 * temperature, np.finfo(float).eps)

    affinity = np.exp(-np.minimum(distances, penalty) / temperature)
    affinity[~finite_offdiag] = 0.0
    np.fill_diagonal(affinity, 0.0)

    if not np.any(affinity > 0.0):
        result = np.ones((n, n), dtype=float)
        result[~finite_offdiag] = 0.0
        np.fill_diagonal(result, 0.0)
        return result

    endpoint_gap = np.zeros((n, n), dtype=float)

    for i in range(n):
        alternatives = distances[i].copy()
        alternatives[i] = penalty
        order = np.argsort(alternatives, kind="stable")

        best = alternatives[order[0]]
        second_best = alternatives[order[1]] if n > 2 else best
        local_scale = max(second_best - best, 0.16 * scale)
        local_scale = max(local_scale, np.finfo(float).eps)

        for j in range(n):
            if i == j or not finite_offdiag[i, j]:
                continue

            if j == order[0]:
                gap = second_best - distances[i, j]
            else:
                gap = best - distances[i, j]

            endpoint_gap[i, j] = np.exp(
                np.clip(gap / local_scale, -20.0, 20.0)
            )

    endpoint_gap = np.sqrt(endpoint_gap * endpoint_gap.T)
    endpoint_gap[~finite_offdiag] = 0.0

    best_bypass = np.full((n, n), penalty, dtype=float)

    for k in range(n):
        candidate = distances[:, k, None] + distances[None, k, :]
        best_bypass = np.minimum(best_bypass, candidate)

    np.fill_diagonal(best_bypass, penalty)

    bypass_advantage = np.maximum(best_bypass - distances, 0.0)
    bypass_scale = max(0.58 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(-bypass_advantage / bypass_scale)
    irreplaceability[~finite_offdiag] = 0.0

    maximum_affinity = max(
        float(np.max(affinity)),
        np.finfo(float).eps,
    )
    base = affinity / maximum_affinity

    score = (
        0.54 * base
        + 0.25 * endpoint_gap
        + 0.21 * irreplaceability
    )
    score[~finite_offdiag] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(56):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.58 * score + 0.42 * balanced
    result[~finite_offdiag] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
