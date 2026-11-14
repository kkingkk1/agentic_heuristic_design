import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    symmetric = np.zeros_like(distances, dtype=float)
    both_finite = finite & finite.T
    symmetric[both_finite] = (
        distances[both_finite] + distances.T[both_finite]
    ) * 0.5

    valid_symmetric = valid & valid.T & both_finite
    if not np.any(valid_symmetric):
        return np.zeros_like(distances, dtype=float)

    positive = symmetric[valid_symmetric & (symmetric > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    normalized = np.zeros_like(symmetric, dtype=float)
    normalized[valid_symmetric] = symmetric[valid_symmetric] / scale
    normalized = np.clip(normalized, 0.0, 8.0)

    distance_score = np.exp(-1.35 * normalized)
    distance_score[~valid_symmetric] = 0.0

    rank_score = np.zeros_like(symmetric, dtype=float)
    for i in range(n):
        row_mask = valid_symmetric[i]
        count = int(np.count_nonzero(row_mask))
        if count == 0:
            continue

        row_values = normalized[i, row_mask]
        order = np.argsort(row_values, kind="stable")
        ranks = np.empty(count, dtype=float)
        ranks[order] = np.arange(count, dtype=float)

        local_rank = 1.0 - ranks / max(count - 1, 1)
        rank_score[i, row_mask] = 0.65 + 0.35 * local_rank

    inverse = np.zeros_like(symmetric, dtype=float)
    inverse[valid_symmetric] = 1.0 / (1.0 + normalized[valid_symmetric])

    row_potential = np.zeros(n, dtype=float)
    col_potential = np.zeros(n, dtype=float)

    for i in range(n):
        row_values = inverse[i, valid_symmetric[i]]
        if row_values.size:
            row_potential[i] = np.mean(row_values)

        col_values = inverse[valid_symmetric[:, i], i]
        if col_values.size:
            col_potential[i] = np.mean(col_values)

    global_potential = 0.5 * (
        row_potential[:, None] + col_potential[None, :]
    )
    global_potential = np.clip(global_potential, 0.0, 1.0)

    potential_score = 0.85 + 0.15 * global_potential

    priors = distance_score * rank_score * potential_score

    feasible_values = priors[valid_symmetric]
    maximum = float(np.max(feasible_values)) if feasible_values.size else 0.0

    if maximum <= 0.0:
        return np.zeros_like(distances, dtype=float)

    priors /= maximum

    exploration_floor = 0.025
    priors[valid_symmetric] = (
        exploration_floor
        + (1.0 - exploration_floor) * priors[valid_symmetric]
    )

    priors[~valid_symmetric] = 0.0
    np.fill_diagonal(priors, 0.0)

    return priors
