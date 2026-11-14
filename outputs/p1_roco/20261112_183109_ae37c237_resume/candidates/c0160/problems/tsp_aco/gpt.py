import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    valid = finite_positive & finite_positive.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    positive_values = d[valid]
    eps = np.finfo(float).eps
    global_scale = float(np.median(positive_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = float(np.max(positive_values))
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(positive_values))
    penalty = max(finite_max, global_scale) * 4.0 + global_scale
    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    k = min(7, n - 1)
    ordered = np.sort(work, axis=1)
    neighborhood = ordered[:, :k]
    local_scale = np.median(neighborhood, axis=1)

    row_has_edge = valid.any(axis=1)
    local_scale = np.where(
        row_has_edge & np.isfinite(local_scale) & (local_scale > 0.0),
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

    clipped_relative = np.clip(relative_distance, 0.0, 60.0)
    distance_term = np.exp(-0.9 * clipped_relative)
    long_edge_penalty = np.exp(
        -0.35 * np.maximum(relative_distance - 1.0, 0.0)
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_strength = np.exp(-ranks / rank_scale)
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    choice_scale = np.maximum(local_scale, eps)
    choice_logits = np.divide(
        -work,
        choice_scale[:, None],
        out=np.full((n, n), -np.inf, dtype=float),
        where=np.isfinite(work),
    )
    choice_logits = np.clip(choice_logits, -60.0, 0.0)
    choice_weight = np.exp(choice_logits)
    choice_weight[~valid] = 0.0

    normalizer = choice_weight.sum(axis=1, keepdims=True)
    choice_probability = np.divide(
        choice_weight,
        normalizer,
        out=np.zeros_like(choice_weight),
        where=normalizer > 0.0,
    )
    mutual_preference = np.sqrt(
        choice_probability * choice_probability.T
    )

    best = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else best
    regret = np.divide(
        np.maximum(second - best, 0.0),
        np.maximum(second, eps),
    )
    regret = np.clip(np.nan_to_num(regret), 0.0, 1.0)
    regret_bonus = 1.0 + 0.30 * (
        regret[:, None] + regret[None, :]
    )

    accessibility = np.sum(
        np.where(valid, np.exp(-np.clip(
            work / choice_scale[:, None],
            0.0,
            60.0,
        )), 0.0),
        axis=1,
    )
    scarcity = 1.0 / np.sqrt(
        1.0 + accessibility[:, None] + accessibility[None, :]
    )
    scarcity_bonus = 1.0 + 0.25 * scarcity

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.35, 1.0)

    score = (
        distance_term
        * (0.48 + 0.32 * bidirectional_rank + 0.20 * mutual_preference)
        * regret_bonus
        * scarcity_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
