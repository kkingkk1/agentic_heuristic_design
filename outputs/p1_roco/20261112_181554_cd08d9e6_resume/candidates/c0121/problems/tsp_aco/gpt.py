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

    d = 0.5 * (distances + distances.T)
    d[~symmetric_valid] = np.inf
    np.fill_diagonal(d, np.inf)

    if not np.any(np.isfinite(d)):
        return scores

    ordered = np.sort(d, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else nearest

    finite_nearest = np.isfinite(nearest)
    nearest_safe = np.where(
        finite_nearest,
        np.maximum(nearest, np.finfo(float).eps),
        1.0,
    )
    second_safe = np.where(
        np.isfinite(second),
        np.maximum(second, np.finfo(float).eps),
        nearest_safe,
    )

    local_scale = np.sqrt(nearest_safe * second_safe)

    regret = np.divide(
        second_safe - nearest_safe,
        second_safe,
        out=np.zeros_like(second_safe),
        where=second_safe > 0,
    )
    regret = np.clip(regret, 0.0, 1.0)

    relative_length = np.divide(
        d,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full_like(d, np.inf),
        where=np.isfinite(d),
    )

    distance_attractiveness = np.exp(
        -0.9 * np.clip(relative_length, 0.0, 60.0)
    )

    ranks = np.full((n, n), np.inf, dtype=float)
    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = row_rank[None, :]

    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    mutual_neighborhood = np.exp(
        -0.6 * (ranks + ranks.T) / rank_scale
    )

    endpoint_regret = 1.0 + 0.5 * (
        regret[:, None] + regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.28 * np.maximum(relative_length - 1.0, 0.0)
    )

    score = (
        distance_attractiveness
        * (0.62 + 0.38 * mutual_neighborhood)
        * endpoint_regret
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
