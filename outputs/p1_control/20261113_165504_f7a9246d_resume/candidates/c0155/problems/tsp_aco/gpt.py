import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return result

    pair = np.where(valid, 0.5 * (d + d.T), np.inf)
    eps = np.finfo(float).eps

    positive = pair[valid & (pair > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = pair[valid]
        global_scale = float(np.median(finite_values)) if finite_values.size else 1.0

    global_scale = max(global_scale, eps)

    local_median = np.full(n, global_scale, dtype=float)
    for i in range(n):
        values = pair[i, valid[i]]
        positive_values = values[values > 0.0]
        if positive_values.size:
            local_median[i] = max(float(np.median(positive_values)), eps)
        elif values.size:
            local_median[i] = max(float(np.median(values)), eps)

    endpoint_scale = np.sqrt(
        local_median[:, None] * local_median[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    relative_cost = pair / endpoint_scale
    relative_cost = np.where(valid, relative_cost, np.inf)

    row_counts = np.sum(valid, axis=1).astype(float)
    row_counts = np.maximum(row_counts, 1.0)

    row_better = np.sum(
        valid & (pair <= pair[:, :, None] if False else valid),
        axis=1,
    )

    endpoint_percentile = np.zeros_like(pair, dtype=float)
    for i in range(n):
        values = pair[i]
        mask = valid[i]
        if not np.any(mask):
            continue
        endpoint_percentile[i, mask] = (
            np.sum(values[mask][None, :] <= values[mask][:, None], axis=1)
            / row_counts[i]
        )

    percentile_affinity = np.exp(
        -1.8 * (
            endpoint_percentile
            + endpoint_percentile.T
        )
    )

    best_two_hop = np.full((n, n), np.inf, dtype=float)
    if n >= 3:
        for k in range(n):
            through_k = pair[:, [k]] + pair[[k], :]
            allowed = valid[:, [k]] & valid[[k], :]
            allowed[:, k] = False
            allowed[k, :] = False
            best_two_hop = np.minimum(
                best_two_hop,
                np.where(allowed, through_k, np.inf),
            )

    finite_two_hop = np.isfinite(best_two_hop) & valid
    shortcut = np.zeros_like(pair, dtype=float)

    if np.any(finite_two_hop):
        denominator = best_two_hop + pair + eps
        margin = (best_two_hop - pair) / denominator
        margin = np.clip(margin, -1.0, 1.0)
        shortcut[finite_two_hop] = 0.5 + 0.5 * margin[finite_two_hop]

    no_alternative = valid & ~finite_two_hop
    shortcut[no_alternative] = 0.5

    scale_factor = np.sqrt(
        np.maximum(relative_cost, 0.0)
        * np.maximum(relative_cost.T, 0.0)
    )
    distance_affinity = np.exp(-np.minimum(scale_factor, 50.0))

    score = (
        distance_affinity
        * percentile_affinity
        * (0.35 + 0.65 * shortcut)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score.astype(float, copy=False)
