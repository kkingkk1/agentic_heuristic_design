import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    raw = distance_matrix.copy()
    raw[~np.isfinite(raw)] = np.inf
    raw[raw < 0] = 0.0
    np.fill_diagonal(raw, np.inf)

    reverse = raw.T
    finite_raw = np.isfinite(raw)
    finite_reverse = np.isfinite(reverse)

    distance = np.full((n, n), np.inf, dtype=float)
    both = finite_raw & finite_reverse
    forward_only = finite_raw & ~finite_reverse
    reverse_only = ~finite_raw & finite_reverse

    distance[both] = 0.5 * (raw[both] + reverse[both])
    distance[forward_only] = raw[forward_only]
    distance[reverse_only] = reverse[reverse_only]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    positive = distance[valid & (distance > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1)
    rank_weight = np.zeros((n, n), dtype=float)
    nearest = np.zeros((n, n), dtype=bool)
    top = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    first = np.full(n, global_scale, dtype=float)
    second = np.full(n, global_scale, dtype=float)
    counts = np.zeros(n, dtype=float)

    k = min(n - 1, max(3, int(np.ceil(1.7 * np.sqrt(n)))))

    for i in range(n):
        candidates = order[i, np.isfinite(distance[i, order[i]])]
        if candidates.size == 0:
            continue

        values = distance[i, candidates]
        counts[i] = candidates.size
        first[i] = values[0]
        second[i] = values[1] if values.size > 1 else values[0]

        rank_weight[i, candidates] = 1.0 / np.sqrt(
            np.arange(1, candidates.size + 1, dtype=float)
        )
        top[i, candidates[:k]] = True
        nearest[i, candidates[0]] = True

        positive_values = values[values > 0]
        if positive_values.size:
            local_scale[i] = float(
                np.median(positive_values[:min(8, positive_values.size)])
            )

    local_scale = np.maximum(local_scale, global_scale * 0.06)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized = distance / np.maximum(pair_scale, eps)
    normalized = np.clip(normalized, 0.0, 50.0)

    scale_ratio = np.clip(global_scale / local_scale, 0.25, 4.0)
    beta = np.sqrt(scale_ratio[:, None] * scale_ratio[None, :])

    logits = -normalized * np.clip(beta, 0.30, 2.6)
    logits[~valid] = -np.inf

    preference = np.zeros((n, n), dtype=float)

    for i in range(n):
        mask = valid[i]
        if not np.any(mask):
            continue

        row = logits[i, mask]
        shifted = row - np.max(row)
        weights = np.exp(shifted)
        total = np.sum(weights)

        if np.isfinite(total) and total > 0:
            preference[i, mask] = weights / total

    mutual_preference = np.sqrt(preference * preference.T)
    mutual_rank = np.sqrt(rank_weight * rank_weight.T)

    mutual_top = (top & top.T).astype(float)
    mutual_nearest = (nearest & nearest.T).astype(float)

    gap_i = np.maximum(second, first)
    gap_j = gap_i[:, None]
    edge_distance = np.maximum(distance, eps)

    regret_i = np.clip(gap_j / edge_distance - 1.0, 0.0, 7.0)
    regret_j = np.clip(gap_i[None, :] / edge_distance - 1.0, 0.0, 7.0)
    scarcity = 1.0 + 0.25 * np.sqrt(regret_i * regret_j)

    support = np.maximum(preference, preference.T)
    support = 0.62 + 0.38 * support

    endpoint_strength = np.maximum(rank_weight, rank_weight.T)
    endpoint_strength = 0.70 + 0.30 * endpoint_strength

    density = np.sqrt(
        np.maximum(counts[:, None], 1.0)
        * np.maximum(counts[None, :], 1.0)
    )
    density = np.sqrt(
        np.clip(density / max(n - 1, 1), 0.45, 1.0)
    )
    density = 0.80 + 0.20 * density

    scores = (
        mutual_preference
        * (0.40 + 0.60 * mutual_rank)
        * support
        * endpoint_strength
        * scarcity
        * density
        * (1.0 + 0.75 * mutual_top)
        * (1.0 + 0.60 * mutual_nearest)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores
