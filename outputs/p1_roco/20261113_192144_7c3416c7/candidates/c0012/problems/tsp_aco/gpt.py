import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return scores

    distances = distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[np.isfinite(distances)]
    if finite_values.size == 0:
        return scores

    positive_values = finite_values[finite_values > 0]
    fallback = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.max(finite_values)), 1.0)
    )

    distances[~np.isfinite(distances)] = fallback * 10.0
    distances = np.maximum(distances, 0.0)

    sorted_distances = np.sort(distances, axis=1)
    nearest_count = max(1, min(n - 1, 4))
    local_scale = np.median(
        sorted_distances[:, :nearest_count],
        axis=1,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    global_scale = float(np.median(distances[~np.eye(n, dtype=bool)]))
    global_scale = max(global_scale, np.finfo(float).eps)

    local_attractiveness = np.exp(
        -distances / local_scale[:, None]
    )
    global_attractiveness = np.exp(
        -distances / global_scale
    )

    order = np.argsort(distances, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks += 1.0
    rank_attractiveness = 1.0 / np.sqrt(ranks)

    scores = (
        0.50 * local_attractiveness
        + 0.30 * global_attractiveness
        + 0.20 * rank_attractiveness
    )

    candidate_limit = max(1, min(n - 1, 5))
    candidate_sets = np.zeros((n, n), dtype=bool)

    for i in range(n):
        nearest = order[i, :candidate_limit]
        candidate_sets[i, nearest] = True

    mutual_candidates = candidate_sets & candidate_sets.T
    scores[mutual_candidates] *= 1.35

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
