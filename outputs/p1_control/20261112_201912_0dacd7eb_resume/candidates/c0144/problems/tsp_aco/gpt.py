import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
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
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)

    positive = finite[finite > 0.0]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else max(fallback, np.finfo(float).eps)
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    ordered = np.sort(d, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint_ratio_a = d / local_scale[:, None]
    endpoint_ratio_b = d / local_scale[None, :]

    endpoint_compatibility = np.exp(
        -0.65
        * (
            np.clip(endpoint_ratio_a, 0.0, 50.0) ** 0.8
            + np.clip(endpoint_ratio_b, 0.0, 50.0) ** 0.8
        )
        / 2.0
    )

    relative_distance = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    distance_term = np.exp(
        -np.clip(relative_distance, 0.0, 50.0) ** 0.9
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_a = row_rank / max(float(n - 1), 1.0)
    rank_b = row_rank.T / max(float(n - 1), 1.0)
    mutual_rank = 0.5 * (rank_a + rank_b)

    reciprocal_bonus = 1.0 + 0.28 * np.exp(
        -2.2 * np.clip(mutual_rank, 0.0, 1.0)
    )

    sparse_gate = 0.20 + 0.80 / (
        1.0
        + np.exp(
            5.0
            * (
                np.clip(mutual_rank, 0.0, 1.0)
                - min(0.75, float(k) / max(float(n - 1), 1.0))
            )
        )
    )

    global_term = np.exp(
        -0.35 * np.clip(d / global_scale, 0.0, 50.0) ** 0.85
    )

    score = (
        endpoint_compatibility
        * distance_term
        * reciprocal_bonus
        * sparse_gate
        * (0.65 + 0.35 * global_term)
    )

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
