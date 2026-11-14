import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0.0)
    valid = positive & positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.zeros((n, n), dtype=float)
    symmetric[valid] = 0.5 * (d[valid] + d.T[valid])

    values = symmetric[valid]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(values)), eps)
    largest = max(float(np.max(values)), global_scale)
    replacement = 12.0 * largest

    work = np.where(valid, symmetric, replacement)
    np.fill_diagonal(work, replacement)

    row_counts = np.sum(valid, axis=1).astype(float)
    row_sums = np.sum(np.where(valid, symmetric, 0.0), axis=1)

    row_mean = np.divide(
        row_sums,
        np.maximum(row_counts, 1.0),
        out=np.full(n, global_scale, dtype=float),
        where=row_counts > 0,
    )
    row_mean = np.maximum(row_mean, eps)

    centered = np.where(valid, symmetric - row_mean[:, None], 0.0)
    row_variance = np.divide(
        np.sum(centered * centered, axis=1),
        np.maximum(row_counts, 1.0),
    )
    row_spread = np.sqrt(row_variance)
    row_temperature = np.maximum(row_mean + 0.5 * row_spread, eps)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logits = -work / row_temperature[:, None]
        logits = np.maximum(logits, -60.0)
        transition = np.exp(logits) * valid

    transition_sum = np.sum(transition, axis=1, keepdims=True)
    transition = np.divide(
        transition,
        np.maximum(transition_sum, eps),
        out=np.zeros_like(transition),
        where=transition_sum > 0.0,
    )

    endpoint_confidence = np.sqrt(
        transition * transition.T
    )

    left_scale = row_temperature[:, None]
    right_scale = row_temperature[None, :]
    pair_scale = np.sqrt(left_scale * right_scale)

    relative_cost = work / np.maximum(pair_scale, eps)
    direct_preference = np.exp(
        -0.85 * np.minimum(relative_cost, 40.0)
    )

    leg_from_left = np.exp(
        -np.minimum(work / np.maximum(row_temperature[:, None], eps), 40.0)
    ) * valid
    leg_from_right = np.exp(
        -np.minimum(work / np.maximum(row_temperature[None, :], eps), 40.0)
    ) * valid

    shared_support = leg_from_left @ leg_from_right.T
    shared_support = np.sqrt(
        np.maximum(shared_support, 0.0)
        * np.maximum(shared_support.T, 0.0)
    )

    support_scale = np.maximum(
        np.sqrt(
            np.sum(leg_from_left, axis=1)[:, None]
            * np.sum(leg_from_right, axis=0)[None, :]
        ),
        eps,
    )
    normalized_support = shared_support / support_scale
    support_preference = 0.72 + 0.28 * np.tanh(2.5 * normalized_support)

    two_hop = leg_from_left @ work
    two_hop = np.minimum(two_hop, two_hop.T)
    two_hop = np.maximum(two_hop, eps)

    direct_to_detour = work / two_hop
    detour_preference = 0.78 + 0.22 * np.exp(
        -0.65 * np.minimum(direct_to_detour, 12.0)
    )

    bridge_factor = 1.0 + 0.10 * np.minimum(
        relative_cost / np.maximum(global_scale / pair_scale, eps),
        3.0,
    )

    score = (
        endpoint_confidence
        * direct_preference
        * support_preference
        * detour_preference
        * bridge_factor
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
