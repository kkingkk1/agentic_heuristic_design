import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    n = distance_matrix.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diagonal = d[~np.eye(n, dtype=bool)]
    positive_distances = off_diagonal[off_diagonal > 0.0]

    if positive_distances.size:
        global_scale = float(np.median(positive_distances))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    scale_floor = max(global_scale * 1e-6, eps)

    masked = d.copy()
    np.fill_diagonal(masked, np.inf)

    sorted_rows = np.sort(masked, axis=1)
    k = min(3, n - 1)
    local_scale = np.maximum(sorted_rows[:, k - 1], scale_floor)

    global_affinity = np.exp(-d / max(global_scale, eps))

    row_affinity = np.exp(-d / local_scale[:, None])
    symmetric_local_affinity = np.sqrt(row_affinity * row_affinity.T)

    strictly_better = d[:, :, None] > d[:, None, :]
    ranks = np.sum(strictly_better, axis=2).astype(float)
    np.fill_diagonal(ranks, n)

    rank_score = 1.0 / (1.0 + ranks)
    symmetric_rank_score = np.sqrt(rank_score * rank_score.T)

    nearest_distance = np.min(masked, axis=1)
    nearest_edges = np.isclose(
        d,
        nearest_distance[:, None],
        rtol=1e-12,
        atol=scale_floor,
    )
    np.fill_diagonal(nearest_edges, False)

    mutual_nearest = nearest_edges & nearest_edges.T
    mutual_bonus = mutual_nearest.astype(float)

    modulation = (
        0.55
        + 0.20 * symmetric_local_affinity
        + 0.15 * symmetric_rank_score
        + 0.10 * mutual_bonus
    )

    score = global_affinity * modulation
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score
