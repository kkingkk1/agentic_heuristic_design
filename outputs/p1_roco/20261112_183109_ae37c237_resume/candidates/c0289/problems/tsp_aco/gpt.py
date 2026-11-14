import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return scale-invariant edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    symmetric = 0.5 * (distances + distances.T)
    symmetric[~valid] = np.inf
    np.fill_diagonal(symmetric, np.inf)

    finite_values = symmetric[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    degree = np.sum(valid, axis=1, dtype=float)
    finite_max = float(np.max(finite_values))
    sentinel = max(finite_max, global_scale) * 4.0 + global_scale

    work = np.where(valid, symmetric, sentinel)
    np.fill_diagonal(work, sentinel)

    nearest = np.partition(work, kth=1, axis=1)[:, :2]
    best = nearest[:, 0]
    second = nearest[:, 1]

    best = np.where(np.isfinite(best) & (best < sentinel), best, global_scale)
    second = np.where(
        (degree >= 2.0) & np.isfinite(second) & (second < sentinel),
        second,
        best,
    )

    local_scale = np.sqrt(
        np.maximum(best, eps) * np.maximum(second, eps)
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    regret = np.divide(
        second - best,
        np.maximum(second, global_scale * eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    nearest_neighbor = np.argmin(work, axis=1)
    indices = np.arange(n)
    mutual = (
        (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
    )

    scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    relative = np.divide(
        symmetric,
        scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (scale > 0.0),
    )

    row_ratio = np.divide(
        symmetric,
        local_scale[:, None],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        symmetric,
        local_scale[None, :],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    distance_term = np.exp(-np.clip(1.15 * relative, 0.0, 50.0))

    endpoint_term = 1.0 / (
        1.0 + 0.5 * np.clip(row_ratio + col_ratio, 0.0, 100.0)
    )

    rank_term = 1.0 / (
        1.0
        + np.sqrt(
            np.clip(row_ratio * col_ratio, 0.0, 1.0e4)
        )
    )

    competition = np.sum(
        valid & (symmetric <= second[:, None]),
        axis=1,
        dtype=float,
    )
    competition = np.maximum(competition, 1.0)

    scarcity = 1.0 + 0.12 / np.sqrt(
        competition[:, None] * competition[None, :]
    )

    regret_bonus = 1.0 + 0.22 * (
        regret[:, None] + regret[None, :]
    )

    mutual_bonus = np.where(mutual, 1.20, 1.0)

    excess = np.maximum(relative - 1.0, 0.0)
    long_edge_penalty = np.exp(-np.clip(0.45 * excess, 0.0, 30.0))

    scores = (
        distance_term
        * (0.56 + 0.26 * endpoint_term + 0.18 * rank_term)
        * regret_bonus
        * mutual_bonus
        * scarcity
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
