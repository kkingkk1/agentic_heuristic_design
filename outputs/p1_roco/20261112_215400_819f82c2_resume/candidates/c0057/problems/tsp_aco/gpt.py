import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    x = d.copy()
    offdiag = ~np.eye(n, dtype=bool)

    finite_positive = x[offdiag & np.isfinite(x) & (x > 0)]
    fallback = (
        float(np.median(finite_positive))
        if finite_positive.size
        else 1.0
    )

    invalid = ~np.isfinite(x) | (x <= 0)
    x[invalid] = fallback * 1e6
    np.fill_diagonal(x, np.inf)

    valid_counts = np.sum(np.isfinite(x), axis=1)
    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))

    ordered = np.sort(x, axis=1)
    nearest = ordered[:, :k]

    local_scale = np.median(nearest, axis=1)
    local_scale[
        ~np.isfinite(local_scale) | (local_scale <= 0)
    ] = fallback

    # Endpoint-normalized distance: an edge must be short for both endpoints.
    endpoint_distance = 0.5 * (
        x / (local_scale[:, None] + 1e-12)
        + x / (local_scale[None, :] + 1e-12)
    )
    local_closeness = np.exp(-np.minimum(endpoint_distance, 60.0))

    # Dense rank agreement rewards edges that are good neighbors from both sides.
    ranks = np.argsort(np.argsort(x, axis=1), axis=1).astype(float)
    rank_fraction = ranks / max(1.0, float(n - 2))
    rank_quality = np.exp(-2.0 * rank_fraction)
    mutual_rank = np.sqrt(rank_quality * rank_quality.T)

    # A weak global-distance term breaks ties without overwhelming local structure.
    finite_edges = x[offdiag & np.isfinite(x)]
    global_scale = (
        float(np.median(finite_edges))
        if finite_edges.size
        else fallback
    )
    global_closeness = np.exp(
        -np.minimum(x / (global_scale + 1e-12), 60.0)
    )

    score = (
        np.power(local_closeness, 0.60)
        * np.power(mutual_rank, 0.30)
        * np.power(global_closeness, 0.10)
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
