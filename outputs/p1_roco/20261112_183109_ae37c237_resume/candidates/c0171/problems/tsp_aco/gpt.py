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

    valid = np.isfinite(distances) & (distances > 0.0)
    valid &= ~np.eye(n, dtype=bool)
    valid &= valid.T

    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_counts = np.sum(np.isfinite(d), axis=1)
    if not np.any(finite_counts > 0):
        return scores

    ordered = np.sort(d, axis=1)
    nearest = ordered[:, 0]

    if n > 2:
        second = ordered[:, 1]
    else:
        second = nearest

    finite_values = d[np.isfinite(d)]
    global_scale = np.median(finite_values)
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    nearest_safe = np.where(
        np.isfinite(nearest),
        np.maximum(nearest, np.finfo(float).eps),
        global_scale,
    )
    second_safe = np.where(
        np.isfinite(second),
        np.maximum(second, nearest_safe),
        nearest_safe,
    )

    local_scale = np.sqrt(nearest_safe * second_safe)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    regret = np.divide(
        second_safe - nearest_safe,
        second_safe,
        out=np.zeros_like(second_safe),
        where=second_safe > 0.0,
    )
    regret = np.clip(regret, 0.0, 1.0)

    relative_length = np.divide(
        d,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full_like(d, np.inf),
        where=np.isfinite(d),
    )

    clipped_relative = np.clip(relative_length, 0.0, 80.0)

    distance_attractiveness = np.exp(
        -0.72 * np.sqrt(clipped_relative)
        -0.20 * clipped_relative
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.full((n, n), np.inf, dtype=float)
    positions = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = positions[None, :]

    rank_scale = max(1.0, 0.72 * np.sqrt(float(n)))
    rank_score = np.exp(
        -0.85 * (ranks + ranks.T) / rank_scale
    )

    mutual_bonus = 0.72 + 0.58 * rank_score

    endpoint_bonus = 1.0 + 0.68 * (
        regret[:, None] + regret[None, :]
    ) / 2.0

    local_bridge_bonus = 1.0 + 0.16 * np.exp(
        -0.45 * np.abs(
            np.log(
                np.maximum(local_scale[:, None], np.finfo(float).eps)
                / np.maximum(local_scale[None, :], np.finfo(float).eps)
            )
        )
    )

    long_edge_penalty = np.exp(
        -0.20 * np.maximum(clipped_relative - 1.0, 0.0)
    )

    score = (
        distance_attractiveness
        * mutual_bonus
        * endpoint_bonus
        * local_bridge_bonus
        * long_edge_penalty
    )

    score[~valid] = 0.0
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
