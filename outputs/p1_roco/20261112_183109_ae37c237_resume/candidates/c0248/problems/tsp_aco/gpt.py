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

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    eps = np.finfo(float).eps
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    upper_scale = max(finite_max, global_scale)
    penalty = 5.0 * upper_scale + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_distances = np.sort(work, axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else best

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        work,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    distance_term = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.85 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.45
            * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.36 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.30, 1.0)

    long_edge_penalty = np.exp(
        -0.72 * np.maximum(relative_distance - 1.0, 0.0)
    )

    valid_rank = np.where(valid, ranks, float(n))
    scarcity = np.clip(
        1.0 - valid_rank / max(1.0, float(n - 1)),
        0.0,
        1.0,
    )

    local_quality = np.exp(
        -0.5 * np.clip(
            work / local_scale[:, None]
            + work / local_scale[None, :],
            0.0,
            60.0,
        )
    )

    quality_gate = np.clip(
        0.5 + 0.5 * local_quality,
        0.5,
        1.0,
    )

    row_median = np.median(
        np.where(valid, work, np.nan),
        axis=1
    )
    row_median = np.nan_to_num(
        row_median,
        nan=global_scale,
        posinf=global_scale,
        neginf=global_scale,
    )
    row_median = np.maximum(row_median, eps)

    bridge_ratio = np.divide(
        work,
        np.sqrt(row_median[:, None] * row_median[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=(row_median[:, None] * row_median[None, :]) > 0.0,
    )

    bridge_signal = np.clip(
        bridge_ratio - 1.0,
        0.0,
        2.0,
    ) / 2.0

    bridge_bonus = 1.0 + 0.16 * bridge_signal * quality_gate

    scarcity_bonus = 1.0 + 0.16 * (
        scarcity * scarcity.T
    ) * quality_gate

    scores = (
        distance_term
        * (
            0.52
            + 0.28 * bidirectional_rank
            + 0.20 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
        * scarcity_bonus
        * bridge_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
