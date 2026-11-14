import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    # Construct a symmetric effective distance matrix.
    forward = np.where(finite, distances, np.inf)
    reverse = forward.T
    both_valid = np.isfinite(forward) & np.isfinite(reverse)

    effective = np.where(
        both_valid,
        0.5 * (forward + reverse),
        np.minimum(forward, reverse),
    )
    valid = np.isfinite(effective) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = effective[valid & (effective > 0.0)]
    finite_values = effective[valid]

    eps = np.finfo(float).eps
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)
    global_scale = max(global_scale, eps)

    safe = np.where(valid, effective, np.inf)

    # Robust local scale from the nearest usable neighborhood.
    neighborhood = min(max(3, int(np.sqrt(n))), n - 1)
    sorted_distances = np.sort(safe, axis=1)
    nearest = sorted_distances[:, :neighborhood]

    finite_nearest = np.isfinite(nearest)
    local_scale = np.divide(
        np.sum(np.where(finite_nearest, nearest, 0.0), axis=1),
        np.maximum(np.sum(finite_nearest, axis=1), 1),
    )
    local_scale = np.where(
        local_scale > eps,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    # Bidirectional local desirability.
    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    distance_ratio = safe / endpoint_scale
    desirability = 1.0 / (1.0 + np.maximum(distance_ratio, 0.0))

    # Reciprocal rank rewards edges that are scarce for both endpoints.
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )
    reciprocal_rank = np.clip(reciprocal_rank, 0.0, 1.0)

    # Reinforce mutual nearest-neighbor edges.
    nearest_neighbor = np.argmin(safe, axis=1)
    mutual = (
        valid
        & (nearest_neighbor[:, None] == np.arange(n)[None, :])
        & (nearest_neighbor[None, :] == np.arange(n)[:, None])
    )

    # Isolated endpoints receive a moderate bonus for their best usable edges.
    isolation_ratio = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) / (safe + global_scale)
    isolation_bonus = 1.0 + 0.20 * np.clip(isolation_ratio, 0.0, 1.0)

    score = (
        desirability ** 1.25
        * (0.55 + 0.45 * reciprocal_rank)
        * isolation_bonus
        * (1.0 + 0.30 * mutual.astype(float))
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
