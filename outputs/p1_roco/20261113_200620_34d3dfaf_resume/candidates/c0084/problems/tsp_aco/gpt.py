import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & off_diagonal

    if not np.any(valid):
        return result

    values = d[valid]
    distance_scale = float(np.median(values))
    if not np.isfinite(distance_scale) or distance_scale <= 0.0:
        positive_values = values[values > 0.0]
        distance_scale = (
            float(np.median(positive_values))
            if positive_values.size
            else 1.0
        )
    distance_scale = max(distance_scale, 1e-12)

    upper_quantile = float(np.quantile(values, 0.95))
    if not np.isfinite(upper_quantile) or upper_quantile <= 0.0:
        upper_quantile = distance_scale
    upper_quantile = max(upper_quantile, distance_scale * 1e-6)

    clipped = np.zeros_like(d, dtype=float)
    clipped[valid] = np.minimum(d[valid], upper_quantile)

    eps = max(distance_scale * 1e-9, 1e-12)
    reciprocal = np.zeros_like(d, dtype=float)
    reciprocal[valid] = 1.0 / (clipped[valid] + eps)

    row_max = np.max(reciprocal, axis=1, keepdims=True)
    row_signal = reciprocal / np.maximum(row_max, eps)

    incoming = reciprocal.T
    column_max = np.max(incoming, axis=1, keepdims=True)
    column_signal = incoming / np.maximum(column_max, eps)

    reciprocal_signal = np.sqrt(row_signal * column_signal)

    row_order = np.argsort(
        np.where(valid, clipped, np.inf),
        axis=1,
        kind="stable",
    )
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    column_order = np.argsort(
        np.where(valid.T, clipped.T, np.inf),
        axis=1,
        kind="stable",
    )
    column_rank = np.full((n, n), float(n), dtype=float)
    column_rank[
        np.arange(n)[:, None],
        column_order,
    ] = np.arange(n, dtype=float)
    column_rank = column_rank.T

    rank_temperature = max(1.0, np.sqrt(float(n)))
    row_rank_signal = np.exp(-row_rank / rank_temperature)
    column_rank_signal = np.exp(-column_rank / rank_temperature)
    bidirectional_rank = np.sqrt(row_rank_signal * column_rank_signal)

    base = (
        0.65 * reciprocal_signal
        + 0.35 * bidirectional_rank
    )
    base[~valid] = 0.0

    exploration_floor = 0.02
    score = (1.0 - exploration_floor) * base
    score += exploration_floor * valid.astype(float)
    score[~valid] = 0.0

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return result
    score /= maximum

    finite_seed_values = np.nan_to_num(
        clipped,
        nan=0.0,
        posinf=upper_quantile,
        neginf=0.0,
    )
    seed_value = (
        np.sum(finite_seed_values) * 1_000_003.0
        + np.sum(
            finite_seed_values
            * np.arange(1, n + 1, dtype=float)[:, None]
        ) * 97.0
        + n * 104_729.0
    )
    seed = int(abs(seed_value)) % (2**63 - 1)
    rng = np.random.default_rng(seed)

    samples = int(np.clip(6 * n, 48, 192))
    candidate_width = int(np.clip(2 * np.sqrt(n), 4, 24))
    counts = np.zeros((n, n), dtype=float)

    for _ in range(samples):
        start = int(rng.integers(n))
        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        while np.any(remaining):
            candidates = np.flatnonzero(remaining & valid[current])

            if candidates.size == 0:
                break

            if candidates.size > candidate_width:
                local_distances = clipped[current, candidates]
                selected = np.argpartition(
                    local_distances,
                    candidate_width - 1,
                )[:candidate_width]
                candidates = candidates[selected]

            logits = np.log(score[current, candidates] + 1e-12)
            logits -= clipped[current, candidates] / distance_scale
            logits /= 0.35
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            probability_sum = float(np.sum(probabilities))
            if not np.isfinite(probability_sum) or probability_sum <= 0.0:
                probabilities = np.full(
                    candidates.size,
                    1.0 / candidates.size,
                    dtype=float,
                )
            else:
                probabilities /= probability_sum

            next_node = int(rng.choice(candidates, p=probabilities))
            counts[current, next_node] += 1.0
            remaining[next_node] = False
            current = next_node

        if valid[current, start]:
            counts[current, start] += 1.0

    counts /= max(float(samples), 1.0)

    result = 0.75 * score + 0.25 * counts
    result[~valid] = 0.0

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return result.astype(float, copy=False)
