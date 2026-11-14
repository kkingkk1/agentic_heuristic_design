import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return lightweight symmetric edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = (
        np.isfinite(distances)
        & (distances > 0.0)
    )
    np.fill_diagonal(finite_positive, False)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(distances[finite_positive]))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
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
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))

    # Partial selection avoids sorting every row while estimating local scale.
    nearest_values = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_weights = np.linspace(1.0, 0.65, k, dtype=float)
    local_weights /= np.sum(local_weights)

    local_scale = nearest_values @ local_weights
    local_scale = np.maximum(local_scale, eps)

    # Distance affinity uses both endpoint scales, preserving symmetry.
    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = d / np.maximum(pair_scale, eps)
    distance_affinity = np.exp(
        -np.minimum(normalized_distance, 50.0)
    )

    # Smooth sparse-candidate support from each endpoint.
    kth_distance = np.max(nearest_values, axis=1)
    threshold_scale = np.maximum(kth_distance, eps)

    candidate_i = np.exp(
        -np.minimum(
            np.maximum(d - threshold_scale[:, None], 0.0)
            / threshold_scale[:, None],
            30.0,
        )
    )
    candidate_j = np.exp(
        -np.minimum(
            np.maximum(d - threshold_scale[None, :], 0.0)
            / threshold_scale[None, :],
            30.0,
        )
    )

    candidate_support = np.sqrt(candidate_i * candidate_j)

    # Small mutual-neighborhood bonus favors edges useful to both endpoints.
    mutual_bonus = 0.90 + 0.10 * candidate_support

    score = distance_affinity * (
        0.72 + 0.28 * candidate_support
    ) * mutual_bonus

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
