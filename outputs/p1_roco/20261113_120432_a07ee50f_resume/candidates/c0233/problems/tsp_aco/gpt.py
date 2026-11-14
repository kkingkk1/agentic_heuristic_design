import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized TSP edge prior."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    # Use only edges valid in both directions.
    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive = distances[valid & (distances > 0.0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite = distances[valid]
        global_scale = max(float(np.median(finite)), 1.0)

    global_scale = max(global_scale, eps)

    symmetric_distances = np.where(
        valid,
        0.5 * (distances + distances.T),
        np.inf,
    )
    np.fill_diagonal(symmetric_distances, np.inf)

    finite_edges = symmetric_distances[np.isfinite(symmetric_distances)]
    if finite_edges.size == 0:
        return result

    # Estimate a robust local scale from each node's nearest usable edges.
    degree = np.sum(np.isfinite(symmetric_distances), axis=1)
    k = min(4, n - 1)

    nearest_values = np.sort(symmetric_distances, axis=1)[:, :k]
    local_scale = np.mean(
        np.where(
            np.isfinite(nearest_values),
            nearest_values,
            global_scale,
        ),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    # Endpoint-local scale using a harmonic mean, which emphasizes
    # edges that are short relative to both incident neighborhoods.
    endpoint_scale = (
        2.0
        * local_scale[:, None]
        * local_scale[None, :]
        / np.maximum(
            local_scale[:, None] + local_scale[None, :],
            eps,
        )
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    finite_mask = np.isfinite(symmetric_distances)

    # Robust global and local shortness terms.
    global_shortness = np.exp(
        -np.clip(symmetric_distances / global_scale, 0.0, 50.0)
    )
    local_shortness = np.exp(
        -np.clip(symmetric_distances / endpoint_scale, 0.0, 50.0)
    )

    # Reciprocal rank rewards edges that are competitive from both endpoints.
    order = np.argsort(symmetric_distances, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    # Mutual nearest neighbors receive a moderate structural bonus.
    nearest = np.argmin(symmetric_distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i, j in enumerate(nearest):
        if np.isfinite(symmetric_distances[i, j]) and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Slightly favor edges connected to sparse endpoints.
    finite_counts = np.maximum(degree.astype(float), 1.0)
    isolation = 1.0 / np.sqrt(finite_counts[:, None] * finite_counts[None, :])
    isolation /= np.max(isolation) if np.max(isolation) > 0.0 else 1.0

    score = (
        np.sqrt(global_shortness * local_shortness)
        * (0.55 + 0.45 * reciprocal_rank)
        * (1.0 + 0.28 * mutual)
        * (1.0 + 0.12 * isolation)
    )

    score[~finite_mask] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
