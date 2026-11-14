import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    distances = np.where(
        np.isfinite(distance_matrix),
        np.maximum(distance_matrix, 0.0),
        np.inf,
    )
    np.fill_diagonal(distances, np.inf)

    valid = np.isfinite(distances)
    finite_values = distances[valid]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    scales = np.percentile(finite_values, [30.0, 55.0, 80.0])
    scales = np.maximum(scales, np.finfo(float).eps)

    # Multi-scale global desirability.
    global_score = np.zeros((n, n), dtype=float)
    for scale in scales:
        normalized = distances / scale
        component = 1.0 / (1.0 + normalized ** 1.8)
        component[~valid] = 0.0
        global_score += component
    global_score /= float(len(scales))

    # Row-wise rank signal, averaged across local and broader neighborhoods.
    order = np.argsort(distances, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rank_positions = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_positions[None, :]

    valid_counts = valid.sum(axis=1).astype(float)
    rank_score = np.zeros((n, n), dtype=float)

    neighborhood_scales = [
        1.0,
        max(2.0, np.sqrt(max(float(n - 1), 1.0))),
        max(3.0, float(n - 1)),
    ]

    for neighborhood in neighborhood_scales:
        component = np.exp(-ranks / neighborhood)
        component[~valid] = 0.0
        rank_score += component
    rank_score /= float(len(neighborhood_scales))

    # Symmetric local-rank support preserves compatibility from both endpoints.
    reverse_rank_score = rank_score.T
    local_rank = 0.5 * (rank_score + reverse_rank_score)

    # Mutual-nearest-neighbor bonus.
    nearest = np.argmin(distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if np.isfinite(distances[i, j]) and nearest[j] == i:
            mutual[i, j] = 1.0

    mutual = np.maximum(mutual, mutual.T)

    # Two-step completion potential:
    # after selecting i -> j, estimate the cheapest j -> k -> i closure.
    two_step = distances[None, :, :] + distances[:, :, None]
    completion = np.min(two_step, axis=0)
    completion_valid = np.isfinite(completion) & valid

    completion_score = np.zeros((n, n), dtype=float)
    for scale in scales:
        normalized = completion / (2.0 * scale)
        component = 1.0 / (1.0 + normalized ** 1.4)
        component[~completion_valid] = 0.0
        completion_score += component
    completion_score /= float(len(scales))

    # Mild bounded long-term bonus; no component can dominate the raw signal.
    scores = (
        0.56 * global_score
        + 0.25 * local_rank
        + 0.11 * completion_score
        + 0.08 * mutual
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = scores.max()
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return scores
