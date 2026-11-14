import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n == 0:
        return scores

    finite = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    undirected_distances = 0.5 * (distances + distances.T)
    edge_distances = np.where(valid, undirected_distances, np.inf)

    row_counts = valid.sum(axis=1)
    finite_values = edge_distances[np.isfinite(edge_distances)]

    global_scale = np.median(finite_values)
    global_scale = max(float(global_scale), 1e-12)

    row_medians = np.full(n, global_scale, dtype=float)
    row_scales = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = edge_distances[i, valid[i]]
        if values.size:
            row_medians[i] = np.median(values)
            nearest = np.partition(values, min(2, values.size - 1))
            row_scales[i] = max(float(nearest[min(2, values.size - 1)]), 1e-12)

    local_scale = np.maximum(row_scales, 1e-12)

    inverse = np.where(valid, 1.0 / np.maximum(edge_distances, 1e-12), 0.0)
    inverse = inverse / np.maximum(row_medians[:, None], 1e-12)

    local_closeness = np.where(
        valid,
        np.exp(-edge_distances / local_scale[:, None]),
        0.0,
    )

    attractive_mass = local_closeness.sum(axis=1)
    scarcity = np.zeros(n, dtype=float)

    positive_mass = attractive_mass > 0
    scarcity[positive_mass] = 1.0 / attractive_mass[positive_mass]

    if np.any(positive_mass):
        low = scarcity[positive_mass].min()
        high = scarcity[positive_mass].max()
        if high > low:
            scarcity = (scarcity - low) / (high - low)
        else:
            scarcity.fill(0.0)

    endpoint_scarcity = scarcity[:, None] + scarcity[None, :]

    nearest_one = np.full(n, np.inf, dtype=float)
    nearest_two = np.full(n, np.inf, dtype=float)

    for i in range(n):
        values = edge_distances[i, valid[i]]
        if values.size:
            ordered = np.partition(values, min(1, values.size - 1))
            nearest_one[i] = ordered[0]
            nearest_two[i] = ordered[min(1, values.size - 1)]

    directed_regret = np.where(
        valid & np.isfinite(nearest_two[:, None]),
        np.maximum(nearest_two[:, None] - edge_distances, 0.0)
        / np.maximum(nearest_two[:, None], 1e-12),
        0.0,
    )

    mutual_nearest = (
        valid
        & (edge_distances <= nearest_one[:, None] + 1e-12)
        & (edge_distances <= nearest_one[None, :] + 1e-12)
    ).astype(float)

    directed_competitiveness = np.where(
        valid,
        local_closeness / np.maximum(local_closeness.sum(axis=1, keepdims=True), 1e-12),
        0.0,
    )

    pair_inverse = inverse + inverse.T
    pair_competitiveness = directed_competitiveness + directed_competitiveness.T
    pair_regret = directed_regret + directed_regret.T

    scores = (
        0.32 * pair_inverse
        + 0.30 * pair_competitiveness
        + 0.18 * endpoint_scarcity
        + 0.15 * pair_regret
        + 0.20 * mutual_nearest
    )

    scores *= valid
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0 and np.isfinite(maximum):
        scores /= maximum

    scores[~np.isfinite(scores)] = 0.0
    return scores.astype(float)
