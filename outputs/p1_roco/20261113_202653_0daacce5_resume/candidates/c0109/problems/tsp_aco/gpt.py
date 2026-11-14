import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d >= 0)
    valid_offdiag = valid.copy()
    np.fill_diagonal(valid_offdiag, False)

    if not np.any(valid_offdiag):
        return np.zeros((n, n), dtype=float)

    values = d[valid_offdiag]
    scale = float(np.median(values))
    if not np.isfinite(scale) or scale <= 0:
        positive = values[values > 0]
        scale = float(np.mean(positive)) if positive.size else 1.0
    scale = max(scale, 1e-12)

    largest = float(np.max(values))
    replacement = largest + 16.0 * scale

    clean = np.where(valid, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-9, 1e-12)
    inverse = 1.0 / (clean + eps)
    inverse[~valid_offdiag] = 0.0

    row_total = np.maximum(inverse.sum(axis=1, keepdims=True), eps)
    col_total = np.maximum(inverse.sum(axis=0, keepdims=True), eps)

    outgoing_share = inverse / row_total
    incoming_share = inverse / col_total
    exclusivity = np.sqrt(outgoing_share * incoming_share)

    valid_cost = np.where(valid_offdiag, clean, replacement)
    width = min(n - 1, max(5, int(2.2 * np.sqrt(n))))

    nearest = np.argpartition(
        valid_cost,
        kth=min(width - 1, n - 1),
        axis=1,
    )[:, :width]

    rank = np.full((n, n), float(n), dtype=float)
    rank_values = np.take_along_axis(clean, nearest, axis=1)
    rank_order = np.argsort(rank_values, axis=1)
    ordered_nearest = np.take_along_axis(nearest, rank_order, axis=1)
    ordered_values = np.take_along_axis(rank_values, rank_order, axis=1)
    rank[np.arange(n)[:, None], ordered_nearest] = np.arange(width, dtype=float)
    rank_score = np.exp(-rank / max(1.0, np.sqrt(float(n))))
    rank_score[~valid_offdiag] = 0.0

    local_scale = np.maximum(
        np.mean(ordered_values, axis=1, keepdims=True),
        scale * 1e-6,
    )
    compatibility = np.zeros((n, n), dtype=float)
    compatibility[
        np.arange(n)[:, None],
        ordered_nearest,
    ] = np.exp(-ordered_values / local_scale)
    compatibility[~valid_offdiag] = 0.0

    distance_score = np.exp(-np.minimum(clean / scale, 60.0))
    distance_score[~valid_offdiag] = 0.0

    base = (
        0.30 * distance_score
        + 0.25 * np.sqrt(outgoing_share * incoming_share)
        + 0.25 * rank_score
        + 0.20 * compatibility
    )
    base[~valid_offdiag] = 0.0
    np.fill_diagonal(base, 0.0)

    base_max = float(np.max(base))
    if not np.isfinite(base_max) or base_max <= 0:
        base = valid_offdiag.astype(float)
    else:
        base /= base_max

    seed_values = np.nan_to_num(
        clean,
        nan=0.0,
        posinf=replacement,
        neginf=0.0,
    )
    weights = np.arange(1, n + 1, dtype=float)
    seed_value = (
        np.sum(seed_values) * 1000003.0
        + np.sum(seed_values * weights[:, None]) * 9176.0
        + np.sum(seed_values * weights[None, :]) * 6151.0
        + n * 104729.0
    )
    seed = int(abs(seed_value)) % (2**63 - 1)
    rng = np.random.default_rng(seed)

    samples = int(np.clip(4 * n, 80, 240))
    temperature = 0.46
    counts = np.zeros((n, n), dtype=float)

    log_base = np.log(base + 1e-12)
    log_distance = -np.minimum(clean / scale, 60.0)

    for _ in range(samples):
        start = int(rng.integers(n))
        tour = np.empty(n, dtype=int)
        tour[0] = start

        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        for position in range(1, n):
            available = np.flatnonzero(remaining)

            preferred = nearest[current]
            candidates = preferred[remaining[preferred]]

            if candidates.size == 0:
                candidates = available
            elif candidates.size > min(width, available.size):
                candidates = candidates[:min(width, available.size)]

            if candidates.size == 0:
                candidates = available

            valid_candidates = valid_offdiag[current, candidates]
            if np.any(valid_candidates):
                candidates = candidates[valid_candidates]

            if candidates.size == 0:
                candidates = available

            remaining_after = remaining.copy()
            remaining_after[candidates] = False

            if np.any(remaining_after):
                future = inverse[candidates][:, remaining_after].sum(axis=1)
                future_max = float(np.max(future))
                if future_max > 0 and np.isfinite(future_max):
                    future /= future_max
                else:
                    future.fill(1.0)
            else:
                future = np.ones(candidates.size, dtype=float)

            close = np.exp(
                -np.minimum(clean[candidates, start] / scale, 60.0)
            )

            logits = (
                1.35 * log_base[current, candidates]
                + 0.95 * log_distance[current, candidates]
                + 0.38 * np.log(future + 1e-12)
                + 0.24 * np.log(close + 1e-12)
            )
            logits /= temperature
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            probability_sum = float(np.sum(probabilities))

            if not np.isfinite(probability_sum) or probability_sum <= 0:
                probabilities.fill(1.0 / candidates.size)
            else:
                probabilities /= probability_sum

            next_node = int(rng.choice(candidates, p=probabilities))
            tour[position] = next_node
            remaining[next_node] = False
            current = next_node

        edges_from = tour
        edges_to = np.roll(tour, -1)
        counts[edges_from, edges_to] += 1.0

    counts /= float(samples)
    result = 0.58 * base + 0.42 * counts

    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
