import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & ~diagonal

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    symmetric_finite = finite | finite.T
    cleaned = np.where(np.isfinite(raw), raw, np.nan)

    for i in range(n):
        for j in range(i + 1, n):
            a = cleaned[i, j]
            b = cleaned[j, i]
            if np.isfinite(a) and np.isfinite(b):
                value = 0.5 * (a + b)
            elif np.isfinite(a):
                value = a
            elif np.isfinite(b):
                value = b
            else:
                value = np.nan
            cleaned[i, j] = value
            cleaned[j, i] = value

    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned = np.maximum(cleaned, 0.0)
    np.fill_diagonal(cleaned, 0.0)

    observed = cleaned[symmetric_finite]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    penalty = maximum + 24.0 * scale

    distances = np.where(symmetric_finite, cleaned, penalty)
    np.fill_diagonal(distances, penalty)

    row_min = np.min(distances, axis=1)
    centered = np.maximum(distances - row_min[:, None], 0.0)

    finite_centered = centered[symmetric_finite]
    positive_centered = finite_centered[finite_centered > 0.0]
    gap_scale = (
        float(np.median(positive_centered))
        if positive_centered.size
        else 0.35 * scale
    )
    gap_scale = max(gap_scale, 0.12 * scale, np.finfo(float).eps)

    temperature = max(
        float(np.median(distances[symmetric_finite])),
        0.30 * scale,
        np.finfo(float).eps,
    )

    affinity = np.exp(-centered / temperature)
    affinity[~symmetric_finite] = 0.0
    np.fill_diagonal(affinity, 0.0)

    rank_score = np.zeros((n, n), dtype=float)
    endpoint_regret = np.zeros((n, n), dtype=float)

    for i in range(n):
        order = np.argsort(distances[i], kind="stable")
        order = order[order != i]

        if order.size == 0:
            continue

        ranks = np.empty(n, dtype=float)
        ranks.fill(float(order.size))
        ranks[order] = np.arange(order.size, dtype=float)

        rank_score[i] = np.exp(-ranks / max(0.35 * order.size, 1.0))
        rank_score[i, i] = 0.0

        best = distances[i, order[0]]
        second = distances[i, order[1]] if order.size > 1 else best
        local_gap = max(second - best, 0.16 * scale, gap_scale)

        regret = np.exp(
            np.clip((second - distances[i]) / local_gap, -12.0, 12.0)
        )
        regret[~symmetric_finite[i]] = 0.0
        regret[i] = 0.0
        endpoint_regret[i] = regret

    endpoint_regret = np.sqrt(
        np.maximum(endpoint_regret, 0.0)
        * np.maximum(endpoint_regret.T, 0.0)
    )

    best_two_hop = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        two_hop = distances[:, k, None] + distances[k, None, :]
        best_two_hop = np.minimum(best_two_hop, two_hop)

    np.fill_diagonal(best_two_hop, penalty)

    bypass_gain = np.maximum(best_two_hop - distances, 0.0)
    bypass_scale = max(0.55 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(-bypass_gain / bypass_scale)
    irreplaceability[~symmetric_finite] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    base = affinity / max(float(np.max(affinity)), np.finfo(float).eps)
    rank_score /= max(float(np.max(rank_score)), np.finfo(float).eps)
    endpoint_regret /= max(
        float(np.max(endpoint_regret)),
        np.finfo(float).eps,
    )

    score = (
        0.48 * base
        + 0.20 * rank_score
        + 0.18 * endpoint_regret
        + 0.14 * irreplaceability
    )

    score[~symmetric_finite] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(64):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.62 * score + 0.38 * balanced
    result[~symmetric_finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum_result = float(np.max(result))
    if np.isfinite(maximum_result) and maximum_result > 0.0:
        result /= maximum_result

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
