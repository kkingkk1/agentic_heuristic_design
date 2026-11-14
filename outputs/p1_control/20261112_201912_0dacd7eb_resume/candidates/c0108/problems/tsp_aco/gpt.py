import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid_input = (
        np.isfinite(distances)
        & (distances > 0.0)
        & off_diagonal
    )

    positive = distances[valid_input]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(valid_input, distances, fallback)
    d = 0.5 * (d + d.T)
    d = np.maximum(d, np.finfo(float).eps)
    np.fill_diagonal(d, np.inf)

    finite_mask = np.isfinite(d)
    if not np.any(finite_mask):
        return np.zeros((n, n), dtype=float)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    ordered = np.sort(d, axis=1)

    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    nearest = ordered[:, 0]
    second_nearest = (
        ordered[:, 1] if n > 2 else ordered[:, 0]
    )
    local_gap = np.maximum(
        second_nearest - nearest,
        np.finfo(float).eps,
    )

    endpoint_ratio_i = d / local_scale[:, None]
    endpoint_ratio_j = d / local_scale[None, :]

    endpoint_affinity = np.exp(
        -np.sqrt(
            np.clip(endpoint_ratio_i, 0.0, 50.0)
            * np.clip(endpoint_ratio_j, 0.0, 50.0)
        )
    )

    bottleneck_affinity = np.minimum(
        np.exp(-np.clip(endpoint_ratio_i, 0.0, 50.0)),
        np.exp(-np.clip(endpoint_ratio_j, 0.0, 50.0)),
    )

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(
        n,
        dtype=float,
    )

    normalized_rank = (
        ranks / max(float(n - 1), 1.0)
        + ranks.T / max(float(n - 1), 1.0)
    ) * 0.5
    rank_term = np.exp(-np.clip(1.25 * normalized_rank, 0.0, 50.0))

    nearest_index = np.argmin(d, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest_index):
        if i != j and nearest_index[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    margin_i = np.clip(
        (second_nearest[:, None] - d)
        / local_gap[:, None],
        0.0,
        1.0,
    )
    margin_j = np.clip(
        (second_nearest[None, :] - d)
        / local_gap[None, :],
        0.0,
        1.0,
    )
    mutual_margin = np.sqrt(margin_i * margin_j)

    candidate_count = max(1, int(np.ceil(0.20 * (n - 1))))
    margin_order = np.argsort(
        -mutual_margin,
        axis=1,
        kind="mergesort",
    )
    margin_rank = np.empty_like(margin_order, dtype=float)
    margin_rank[
        np.arange(n)[:, None],
        margin_order,
    ] = np.arange(n, dtype=float)

    high_margin = (
        margin_rank < candidate_count
    ) & off_diagonal
    concentrated_bonus = np.where(
        high_margin & high_margin.T,
        0.10 * mutual_margin,
        0.0,
    )

    score = (
        endpoint_affinity
        * np.sqrt(bottleneck_affinity)
        * rank_term
        * (1.0 + concentrated_bonus)
        * (1.0 + 0.08 * mutual_nearest)
    )

    score[~off_diagonal] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
