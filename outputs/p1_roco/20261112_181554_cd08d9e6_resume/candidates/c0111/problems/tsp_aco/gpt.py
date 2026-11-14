import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, scale-invariant edge-priority matrix for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances > 0)
    valid &= ~np.eye(n, dtype=bool)

    symmetric_valid = valid & valid.T
    if not np.any(symmetric_valid):
        return scores

    distances = 0.5 * (distances + distances.T)
    distances[~symmetric_valid] = np.inf
    np.fill_diagonal(distances, np.inf)

    if not np.any(np.isfinite(distances)):
        return scores

    neighbor_count = min(5, n - 1)
    nearest_values = np.partition(
        distances,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    finite_neighbors = np.isfinite(nearest_values)
    finite_counts = finite_neighbors.sum(axis=1)

    safe_values = np.where(
        finite_neighbors,
        nearest_values,
        np.nan,
    )

    local_scale = np.nanmedian(safe_values, axis=1)
    fallback_scale = np.nanmedian(
        distances[np.isfinite(distances)]
    )

    if not np.isfinite(fallback_scale) or fallback_scale <= 0:
        fallback_scale = 1.0

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback_scale,
    )

    nearest = nearest_values[:, 0]
    second_nearest = (
        nearest_values[:, 1]
        if neighbor_count > 1
        else nearest
    )

    nearest = np.where(np.isfinite(nearest), nearest, local_scale)
    second_nearest = np.where(
        np.isfinite(second_nearest),
        second_nearest,
        nearest,
    )

    regret = np.divide(
        second_nearest - nearest,
        np.maximum(second_nearest, np.finfo(float).eps),
    )
    regret = np.clip(regret, 0.0, 1.0)

    relative_i = np.divide(
        distances,
        local_scale[:, None],
        out=np.full_like(distances, np.inf),
        where=np.isfinite(distances),
    )
    relative_j = np.divide(
        distances,
        local_scale[None, :],
        out=np.full_like(distances, np.inf),
        where=np.isfinite(distances),
    )

    endpoint_relative_length = 0.5 * (relative_i + relative_j)

    distance_score = np.exp(
        -np.clip(endpoint_relative_length, 0.0, 50.0) / 1.25
    )

    ranks = np.full((n, n), np.inf, dtype=float)
    row_order = np.argsort(distances, axis=1)
    row_ranks = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = row_ranks[None, :]

    rank_temperature = max(1.0, np.sqrt(float(n)))
    mutual_rank_score = np.exp(
        -0.5 * (ranks + ranks.T) / rank_temperature
    )

    endpoint_urgency = 1.0 + 0.55 * (
        regret[:, None] + regret[None, :]
    )

    density_balance = np.sqrt(
        np.divide(
            local_scale[:, None],
            np.maximum(local_scale[None, :], np.finfo(float).eps),
        )
    )
    density_balance = np.minimum(density_balance, 1.0 / np.maximum(density_balance, 1e-12))
    density_balance = 0.75 + 0.25 * density_balance

    long_edge_penalty = np.exp(
        -0.20 * np.maximum(endpoint_relative_length - 1.0, 0.0)
    )

    score = (
        distance_score
        * (0.55 + 0.45 * mutual_rank_score)
        * endpoint_urgency
        * density_balance
        * long_edge_penalty
    )

    score[~symmetric_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
