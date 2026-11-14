import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric local-competition prior for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    safe_d = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        valid_values = d[valid]
        global_scale = float(np.median(valid_values)) if valid_values.size else 1.0

    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    eps = np.finfo(float).eps

    # Estimate each endpoint's characteristic neighborhood distance.
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    ordered = np.sort(safe_d, axis=1)
    neighborhood = ordered[:, :k]
    finite_neighborhood = np.isfinite(neighborhood)

    counts = finite_neighborhood.sum(axis=1)
    sums = np.where(finite_neighborhood, neighborhood, 0.0).sum(axis=1)

    local_scale = np.divide(
        sums,
        np.maximum(counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, eps)

    # Shortness relative to the two endpoint neighborhoods.
    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    shortness = np.exp(
        -np.minimum(safe_d / (pair_scale + eps), 50.0)
    )

    # Local contrast: reward edges close to each endpoint's best alternatives.
    nearest = ordered[:, 0]
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    endpoint_contrast = 1.0 / (
        1.0
        + np.maximum(safe_d - nearest[:, None], 0.0)
        / (local_scale[:, None] + global_scale * 1e-6)
    )
    endpoint_contrast = np.sqrt(
        endpoint_contrast * endpoint_contrast.T
    )

    # Smooth rank preference from both endpoints.
    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_preference = np.exp(
        -(ranks + ranks.T) / (2.0 * rank_scale)
    )

    # Reinforce edges that are nearest neighbors in both directions.
    nearest_index = np.argmin(safe_d, axis=1)
    mutual = (
        nearest_index[:, None] == np.arange(n)[None, :]
    ) & (
        nearest_index[None, :] == np.arange(n)[:, None]
    )
    mutual_bonus = 1.0 + 0.30 * mutual.astype(float)

    score = (
        shortness
        * (0.55 + 0.45 * endpoint_contrast)
        * (0.65 + 0.35 * rank_preference)
        * mutual_bonus
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
