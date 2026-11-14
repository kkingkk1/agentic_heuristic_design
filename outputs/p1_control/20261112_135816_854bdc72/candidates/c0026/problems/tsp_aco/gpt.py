import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    d = np.maximum(d, 0.0)
    d = (d + d.T) * 0.5
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(finite_values) * 10.0 + 1.0)
    work_d = np.where(np.isfinite(d), d, fallback)

    row_scale = np.median(
        np.partition(work_d, min(2, n - 1) - 1, axis=1)[:, : min(2, n - 1)],
        axis=1,
    )
    row_scale = np.maximum(row_scale, np.finfo(float).eps)

    relative_distance = work_d / np.sqrt(
        row_scale[:, None] * row_scale[None, :]
    )
    local_affinity = 1.0 / (1.0 + relative_distance)

    order = np.argsort(work_d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values[None, :]
    ranks = ranks / max(1.0, n - 2.0)
    np.fill_diagonal(ranks, np.inf)

    rank_score = 1.0 / (1.0 + np.maximum(ranks, 0.0))
    mutual_neighborhood = np.sqrt(rank_score * rank_score.T)

    nearest = order[:, 0]
    nearest_distance = work_d[np.arange(n), nearest]
    regret = np.maximum(
        (work_d - nearest_distance[:, None])
        / (row_scale[:, None] + nearest_distance[:, None] + 1e-12),
        0.0,
    )
    regret_score = 1.0 / (1.0 + regret + regret.T)

    second_nearest = order[:, min(1, n - 1)]
    second_distance = work_d[np.arange(n), second_nearest]
    bridge_value = (
        nearest_distance[:, None] + nearest_distance[None, :]
    ) / (work_d + nearest_distance[:, None] + nearest_distance[None, :] + 1e-12)
    bridge_value = np.clip(bridge_value, 0.0, 1.0)

    score = (
        0.42 * local_affinity
        + 0.28 * mutual_neighborhood
        + 0.20 * regret_score
        + 0.10 * bridge_value
    )

    score = (score + score.T) * 0.5
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
