import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return scores

    d = np.where(valid, distances, np.inf)
    d = np.minimum(d, d.T)
    valid = np.isfinite(d)
    np.fill_diagonal(valid, False)
    np.fill_diagonal(d, np.inf)

    if not np.any(valid):
        return scores

    finite = d[valid]
    positive = finite[finite > 0.0]
    scale = (
        float(np.median(positive))
        if positive.size
        else max(float(np.max(finite)), 1.0)
    )
    scale = max(scale, np.finfo(float).eps)

    penalty = max(float(np.max(finite)), scale) * 4.0 + scale
    work = np.where(valid, d, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    sorted_distances = np.sort(work, axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1]

    local_scale = 0.5 * (best + second)
    local_scale = np.maximum(local_scale, scale * 1e-6)

    relative = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    distance_affinity = np.exp(-np.clip(relative, 0.0, 60.0))

    rank_affinity = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    endpoint_access = np.exp(
        -np.clip(
            0.5
            * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = (second - best) / np.maximum(second, scale * 1e-6)
    regret = np.clip(regret, 0.0, 1.0)

    nearest = order[:, 0]
    mutual = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )

    scores = (
        0.62 * distance_affinity
        + 0.23 * rank_affinity
        + 0.15 * endpoint_access
    )

    scores *= 1.0 + 0.08 * (
        regret[:, None] + regret[None, :]
    )

    scores *= np.where(mutual, 1.08, 1.0)
    scores[~valid] = 0.0

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
