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

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    observed = raw[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_values = observed[np.isfinite(observed)]
    upper = float(np.quantile(finite_values, 0.95)) if finite_values.size else scale
    penalty = max(upper + 8.0 * scale, scale)

    distances = np.where(valid, np.maximum(raw, 0.0), penalty)
    np.fill_diagonal(distances, penalty)

    valid_values = distances[valid]
    q25, q75 = np.quantile(valid_values, [0.25, 0.75])
    robust_span = max(float(q75 - q25), 0.25 * scale, np.finfo(float).eps)

    # Main signal: robust inverse-distance affinity.
    normalized = np.maximum(distances - q25, 0.0) / robust_span
    affinity = np.exp(-np.clip(normalized, 0.0, 30.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    # Directed local competition: reward edges that are strong from both
    # their source and destination perspectives.
    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_indices = np.flatnonzero(valid[i])
        if row_indices.size:
            order = row_indices[
                np.argsort(distances[i, row_indices], kind="stable")
            ]
            ranks = np.empty(order.size, dtype=float)
            ranks[np.arange(order.size)] = np.arange(order.size)
            row_rank[i, order] = np.exp(-ranks / max(1.0, 0.35 * order.size))

    for j in range(n):
        col_indices = np.flatnonzero(valid[:, j])
        if col_indices.size:
            order = col_indices[
                np.argsort(distances[ col_indices, j], kind="stable")
            ]
            ranks = np.empty(order.size, dtype=float)
            ranks[np.arange(order.size)] = np.arange(order.size)
            col_rank[order, j] = np.exp(-ranks / max(1.0, 0.35 * order.size))

    competition = np.sqrt(row_rank * col_rank)

    # Targeted 2-opt-style correction. An edge is penalized only when a
    # two-hop replacement is substantially cheaper than keeping that edge.
    best_two_hop = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        if k >= n:
            break
        candidate = distances[:, [k]] + distances[[k], :]
        best_two_hop = np.minimum(best_two_hop, candidate)

    np.fill_diagonal(best_two_hop, penalty)
    improvement = distances - best_two_hop
    meaningful = improvement > max(0.35 * robust_span, 0.10 * scale)

    correction = np.zeros((n, n), dtype=float)
    correction[meaningful] = 1.0 - np.exp(
        -np.maximum(improvement[meaningful], 0.0) / robust_span
    )
    correction[~valid] = 0.0
    np.fill_diagonal(correction, 0.0)

    # Keep the correction conservative so promising edges are preserved.
    score = (
        0.52 * affinity
        + 0.38 * competition
        + 0.10 * (affinity * competition)
    )
    score *= 1.0 - 0.24 * correction

    # Small exploration floor over valid edges only.
    positive_score = score[valid]
    floor = (
        0.025 * float(np.median(positive_score))
        if positive_score.size
        else 0.0
    )
    score[valid] = np.maximum(score[valid], floor)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
