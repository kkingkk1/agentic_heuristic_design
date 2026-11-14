import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    pair_valid = valid & valid.T
    if not np.any(pair_valid):
        return scores

    d = np.full((n, n), np.inf, dtype=float)
    d[pair_valid] = 0.5 * (
        distances[pair_valid] + distances.T[pair_valid]
    )
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    finite = np.isfinite(d)

    ordered = np.sort(d, axis=1)
    nearest = ordered[:, 0]

    if n > 2:
        second = ordered[:, 1]
    else:
        second = nearest.copy()

    finite_nearest = np.isfinite(nearest)
    safe_nearest = np.where(
        finite_nearest,
        np.maximum(nearest, eps),
        1.0,
    )
    safe_second = np.where(
        np.isfinite(second),
        np.maximum(second, eps),
        2.0 * safe_nearest,
    )

    local_scale = np.sqrt(safe_nearest * safe_second)
    local_scale = np.maximum(local_scale, eps)

    regret = np.divide(
        safe_second - safe_nearest,
        safe_second,
        out=np.zeros_like(safe_second),
        where=safe_second > 0.0,
    )
    regret = np.clip(regret, 0.0, 1.0)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=finite,
    )

    proximity = np.exp(
        -1.05 * np.clip(relative_distance, 0.0, 60.0)
    )

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(
        n,
        dtype=float,
    )[None, :]

    rank_scale = max(1.0, 0.82 * np.sqrt(float(n)))
    rank_quality = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )

    nearest_neighbor = order[:, 0]
    mutual_neighbor = (
        (nearest_neighbor[:, None] == np.arange(n)[None, :])
        & (nearest_neighbor[None, :] == np.arange(n)[:, None])
    )
    mutual_bonus = np.where(mutual_neighbor, 1.38, 1.0)

    endpoint_quality = np.exp(
        -np.clip(
            d / local_scale[:, None],
            0.0,
            60.0,
        )
    )

    accessibility = np.sum(
        np.where(finite, endpoint_quality, 0.0),
        axis=1,
    )
    scarcity = 1.0 / np.sqrt(
        1.0
        + accessibility[:, None]
        + accessibility[None, :]
    )

    regret_bonus = 1.0 + 0.40 * (
        regret[:, None] + regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.58 * np.maximum(relative_distance - 1.0, 0.0)
    )

    score = (
        proximity
        * (0.48 + 0.32 * rank_quality + 0.20 * endpoint_quality)
        * (1.0 + 0.22 * scarcity)
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    score[~pair_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
