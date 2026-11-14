import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(raw)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.where(finite, np.maximum(raw, 0.0), 0.0)
    observed = distances[valid]

    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_max = float(np.max(observed)) if observed.size else scale
    penalty = max(finite_max + 8.0 * scale, scale)

    safe = np.where(valid, distances, penalty)
    np.fill_diagonal(safe, penalty)

    positive_safe = safe[(safe > 0.0) & (~np.eye(n, dtype=bool))]
    temperature = (
        float(np.median(positive_safe)) if positive_safe.size else scale
    )
    temperature = max(0.26 * temperature, np.finfo(float).eps)

    def row_normalize(values: np.ndarray) -> np.ndarray:
        values = np.where(valid, values, 0.0)
        maxima = np.max(values, axis=1, keepdims=True)
        maxima = np.maximum(maxima, np.finfo(float).eps)
        return np.where(valid, values / maxima, 0.0)

    # Strong local distance affinity.
    reciprocal = scale / (safe + scale)
    reciprocal = row_normalize(reciprocal)

    exponential = np.exp(-np.minimum(safe, penalty) / temperature)
    exponential = row_normalize(exponential)

    # Outgoing and incoming local competition signals.
    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(distances[i, indices], kind="stable")]
            decay = max(1.0, 0.55 * float(order.size))
            outgoing_rank[i, order] = np.exp(
                -np.arange(order.size, dtype=float) / decay
            )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[
                np.argsort(distances[indices, j], kind="stable")
            ]
            decay = max(1.0, 0.55 * float(order.size))
            incoming_rank[order, j] = np.exp(
                -np.arange(order.size, dtype=float) / decay
            )

    outgoing_rank = row_normalize(outgoing_rank)
    incoming_rank = row_normalize(incoming_rank)
    competition = np.sqrt(outgoing_rank * incoming_rank)

    # Bounded two-step lookahead: reward edges that are difficult to replace
    # by a valid two-edge route through an intermediate node.
    best_bypass = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        pair_valid = valid[:, k, None] & valid[None, k, :]
        candidate = distances[:, k, None] + distances[None, k, :]
        best_bypass = np.minimum(
            best_bypass,
            np.where(pair_valid, candidate, np.inf),
        )

    bypass_exists = np.isfinite(best_bypass) & valid
    advantage = np.zeros((n, n), dtype=float)
    advantage[bypass_exists] = (
        best_bypass[bypass_exists] - distances[bypass_exists]
    )

    lookahead = np.zeros((n, n), dtype=float)
    lookahead[bypass_exists] = 1.0 / (
        1.0
        + np.exp(
            -np.clip(advantage[bypass_exists] / max(0.72 * scale, 1e-12), -8.0, 8.0)
        )
    )
    lookahead = row_normalize(lookahead)

    # Mild bridge-diversity signal from both endpoint perspectives.
    bridge = np.sqrt(exponential * competition)
    bridge = row_normalize(bridge)

    score = (
        0.34 * exponential
        + 0.24 * reciprocal
        + 0.20 * competition
        + 0.14 * lookahead
        + 0.08 * bridge
    )

    # Preserve a small amount of probability mass on every valid edge.
    score = np.where(valid, 0.985 * score + 0.015, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score
