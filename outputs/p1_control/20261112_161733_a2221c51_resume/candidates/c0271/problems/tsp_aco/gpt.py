import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge indicators based on local transition probabilities and detour gain."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (d[valid] + d.T[valid])
    np.fill_diagonal(symmetric, np.inf)

    edge_values = symmetric[valid]
    reference = max(float(np.median(edge_values)), eps)

    # Row-wise temperature captures how concentrated each node's choices are.
    row_temperatures = np.full(n, reference, dtype=float)
    for i in range(n):
        values = symmetric[i, valid[i]]
        if values.size:
            row_temperatures[i] = max(
                float(np.median(values)),
                eps,
            )

    safe_temperature = np.maximum(row_temperatures, eps)

    with np.errstate(over="ignore", invalid="ignore"):
        logits = -symmetric / safe_temperature[:, None]

    logits[~valid] = -np.inf
    row_max = np.max(logits, axis=1, keepdims=True)
    stabilized = logits - row_max

    with np.errstate(over="ignore", invalid="ignore"):
        transition = np.exp(stabilized)

    transition[~valid] = 0.0
    row_sum = np.sum(transition, axis=1, keepdims=True)
    transition = np.divide(
        transition,
        np.maximum(row_sum, eps),
        out=np.zeros_like(transition),
        where=row_sum > 0.0,
    )

    # A symmetric endpoint agreement signal.
    endpoint_agreement = np.sqrt(
        np.maximum(transition, 0.0)
        * np.maximum(transition.T, 0.0)
    )

    # Compute the best two-hop alternative for every pair.
    safe_distance = np.where(valid, symmetric, np.inf)
    two_hop = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        through_k = safe_distance[:, k, None] + safe_distance[None, k, :]
        two_hop = np.minimum(two_hop, through_k)

    direct = np.where(valid, symmetric, np.inf)

    # Positive gain means the direct edge beats its best two-hop substitute.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        detour_gain = (two_hop - direct) / np.maximum(two_hop, eps)

    detour_gain = np.nan_to_num(
        detour_gain,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    detour_gain = np.clip(detour_gain, 0.0, 1.0)

    # Retain a small baseline so that structurally necessary longer edges
    # are not deterministically removed by the stochastic sampler.
    necessity = 0.20 + 0.80 * detour_gain
    necessity = 0.5 * (necessity + necessity.T)

    # Mildly reward edges selected by both endpoints relative to their
    # complete local probability distributions.
    mutual_choice = np.sqrt(
        np.maximum(transition * transition.T, 0.0)
    )

    score = endpoint_agreement * mutual_choice * necessity
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
