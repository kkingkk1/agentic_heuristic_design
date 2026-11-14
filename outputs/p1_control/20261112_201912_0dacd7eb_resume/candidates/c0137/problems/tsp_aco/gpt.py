import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)

    positive_values = finite_values[finite_values > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else fallback
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    mutual_limit = min(n - 1, 2 * k)

    ordered = np.sort(d, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint_ratio_a = d / local_scale[:, None]
    endpoint_ratio_b = d / local_scale[None, :]

    relative = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_term = np.exp(
        -np.clip(relative ** 0.85, 0.0, 50.0)
    )

    endpoint_affinity_a = np.exp(
        -np.clip(endpoint_ratio_a ** 0.8, 0.0, 50.0)
    )
    endpoint_affinity_b = np.exp(
        -np.clip(endpoint_ratio_b ** 0.8, 0.0, 50.0)
    )
    worst_endpoint_affinity = np.minimum(
        endpoint_affinity_a,
        endpoint_affinity_b,
    )
    worst_endpoint_affinity = (
        0.35 + 0.65 * worst_endpoint_affinity
    ) ** 1.15

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    normalized_rank_a = row_rank / max(float(n - 1), 1.0)
    normalized_rank_b = row_rank.T / max(float(n - 1), 1.0)

    rank_sum = normalized_rank_a + normalized_rank_b
    rank_term = 1.0 / (1.0 + 0.75 * rank_sum)

    mutual_top = (
        (row_rank < float(mutual_limit))
        & (row_rank.T < float(mutual_limit))
    )
    mutual_rank_bonus = np.ones((n, n), dtype=float)
    mutual_rank_bonus[mutual_top] += 0.15 * np.exp(
        -2.2 * (
            normalized_rank_a[mutual_top]
            + normalized_rank_b[mutual_top]
        )
    )

    local_mismatch = (
        np.abs(np.log1p(endpoint_ratio_a))
        + np.abs(np.log1p(endpoint_ratio_b))
    )
    compatibility_term = np.exp(
        -np.clip(0.75 * local_mismatch, 0.0, 50.0)
    )

    mutual_locality = np.exp(
        -np.clip(
            0.45 * (endpoint_ratio_a + endpoint_ratio_b),
            0.0,
            50.0,
        )
    )
    local_margin_term = 0.45 + 0.55 * mutual_locality

    global_term = np.exp(
        -np.clip((d / global_scale) ** 0.9, 0.0, 50.0)
    )
    global_term = 0.35 + 0.65 * global_term

    score = (
        relative_term
        * worst_endpoint_affinity
        * rank_term
        * compatibility_term
        * local_margin_term
        * global_term
        * mutual_rank_bonus
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
