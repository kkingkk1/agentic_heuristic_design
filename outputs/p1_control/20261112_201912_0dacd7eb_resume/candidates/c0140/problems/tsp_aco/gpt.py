import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
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

    finite_mask = np.isfinite(d)
    if not np.any(finite_mask):
        return np.zeros((n, n), dtype=float)

    finite_values = d[finite_mask]
    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite_mask, d, fallback)

    positive_values = finite_values[finite_values > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else fallback
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    k = min(n - 1, max(2, int(np.sqrt(n)) + 1))
    nearest = np.partition(d, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint_ratio_a = d / local_scale[:, None]
    endpoint_ratio_b = d / local_scale[None, :]
    relative = d / np.sqrt(local_scale[:, None] * local_scale[None, :])

    quality_relative = np.exp(
        -np.clip(relative ** 0.85, 0.0, 50.0)
    )

    endpoint_affinity_a = np.exp(
        -np.clip(endpoint_ratio_a ** 0.8, 0.0, 50.0)
    )
    endpoint_affinity_b = np.exp(
        -np.clip(endpoint_ratio_b ** 0.8, 0.0, 50.0)
    )
    endpoint_affinity = np.minimum(
        endpoint_affinity_a,
        endpoint_affinity_b,
    )

    # Retain flexibility for edges that may be structurally necessary.
    endpoint_affinity = (
        0.35 + 0.65 * endpoint_affinity
    ) ** 1.15

    local_mismatch = (
        np.abs(np.log1p(endpoint_ratio_a))
        + np.abs(np.log1p(endpoint_ratio_b))
    )
    compatibility = np.exp(
        -np.clip(0.75 * local_mismatch, 0.0, 50.0)
    )

    mutual_locality = np.exp(
        -np.clip(
            0.45 * (endpoint_ratio_a + endpoint_ratio_b),
            0.0,
            50.0,
        )
    )
    local_margin = 0.45 + 0.55 * mutual_locality

    global_term = np.exp(
        -np.clip((d / global_scale) ** 0.9, 0.0, 50.0)
    )
    global_term = 0.35 + 0.65 * global_term

    # Efficient approximate top-neighbor membership from the speed candidate.
    rank_limit = min(n - 1, max(4, 2 * k))
    nearest_indices = np.argpartition(
        d,
        rank_limit - 1,
        axis=1,
    )[:, :rank_limit]

    neighbor_mask = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    neighbor_mask[rows, nearest_indices] = 1.0
    neighbor_mask = 0.5 * (neighbor_mask + neighbor_mask.T)

    speed_relative = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    speed_local = 1.0 / (1.0 + speed_relative)
    speed_rank = 0.7 + 0.3 * neighbor_mask
    speed_global = 0.8 + 0.2 / (1.0 + d / global_scale)

    quality_score = (
        quality_relative
        * endpoint_affinity
        * compatibility
        * local_margin
        * global_term
    )

    quality_score *= 1.0 + 0.12 * np.exp(
        -2.2 * (
            endpoint_ratio_a / (1.0 + endpoint_ratio_a)
            + endpoint_ratio_b / (1.0 + endpoint_ratio_b)
        )
    )

    speed_score = speed_local * speed_rank * speed_global

    score = np.sqrt(
        np.maximum(quality_score, 0.0)
        * np.maximum(speed_score, 0.0)
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
