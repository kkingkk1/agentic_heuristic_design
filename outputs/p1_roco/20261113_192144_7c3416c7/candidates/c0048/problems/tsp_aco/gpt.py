import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_off = np.isfinite(d) & off_diagonal
    valid_values = d[finite_off & (d >= 0)]

    if valid_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = valid_values[valid_values > 0]
    scale = float(
        np.median(positive_values)
        if positive_values.size
        else np.median(valid_values)
    )
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    maximum = float(np.max(valid_values))
    replacement = maximum + 10.0 * scale
    clean = np.where(
        finite_off & (d >= 0),
        d,
        replacement,
    ).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(np.finfo(float).eps, scale * 1e-10)

    row_median = np.median(clean, axis=1)
    row_median = np.maximum(row_median, eps)

    local_temperature = np.maximum(
        0.05 * scale,
        0.5 * row_median,
    )

    local_preference = np.exp(
        -np.minimum(clean / local_temperature[:, None], 60.0)
    )
    np.fill_diagonal(local_preference, 0.0)

    row_sum = np.maximum(local_preference.sum(axis=1, keepdims=True), eps)
    row_probability = local_preference / row_sum

    incoming_sum = np.maximum(
        local_preference.sum(axis=0, keepdims=True),
        eps,
    )
    column_probability = local_preference / incoming_sum

    mutual_support = np.sqrt(
        np.maximum(row_probability, 0.0)
        * np.maximum(column_probability, 0.0)
    )

    global_preference = np.exp(
        -np.minimum(clean / max(scale, eps), 60.0)
    )
    np.fill_diagonal(global_preference, 0.0)

    order = np.argsort(clean, axis=1)
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_support = 1.0 / np.sqrt(ranks + 1.0)
    np.fill_diagonal(rank_support, 0.0)

    row_mass = np.maximum(global_preference.sum(axis=1), eps)
    scarcity = 1.0 / row_mass
    scarcity /= max(float(np.mean(scarcity)), eps)
    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )

    base = (
        0.42 * global_preference
        + 0.28 * local_preference
        + 0.18 * mutual_support
        + 0.12 * rank_support
    )
    base *= 0.75 + 0.25 * np.minimum(endpoint_scarcity, 2.0)
    base = 0.5 * (base + base.T)
    base[~finite_off] = 0.0
    np.fill_diagonal(base, 0.0)

    maximum_base = float(np.max(base))
    if maximum_base > 0.0 and np.isfinite(maximum_base):
        base /= maximum_base
    else:
        base = np.where(off_diagonal & finite_off, 1.0, 0.0)

    seed_data = np.nan_to_num(clean / max(scale, eps), nan=0.0)
    seed_value = (
        np.sum(seed_data)
        + 0.37 * np.sum(seed_data * np.arange(1, n + 1)[:, None])
        + 0.19 * np.sum(seed_data * np.arange(1, n + 1)[None, :])
        + 104729 * n
    )
    seed = int(abs(seed_value) % (2**63 - 1))
    rng = np.random.default_rng(seed)

    samples = int(np.clip(8 * n, 64, 256))
    candidate_width = int(np.clip(2 * np.sqrt(n), 4, 24))
    sampling_temperature = 0.45
    counts = np.zeros((n, n), dtype=float)

    for _ in range(samples):
        start = int(rng.integers(n))
        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        while np.any(remaining):
            candidates = np.flatnonzero(remaining)

            if candidates.size > candidate_width:
                nearest = np.argpartition(
                    clean[current, candidates],
                    candidate_width - 1,
                )[:candidate_width]
                candidates = candidates[nearest]

            logits = np.log(base[current, candidates] + 1e-12)
            logits -= clean[current, candidates] / max(scale, eps)
            logits /= sampling_temperature
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

        counts[current, start] += 1.0

    counts /= float(samples)
    counts = 0.5 * (counts + counts.T)
    counts[~finite_off] = 0.0
    np.fill_diagonal(counts, 0.0)

    result = 0.58 * base + 0.42 * counts
    result[~finite_off] = 0.0
    np.fill_diagonal(result, 0.0)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    maximum_result = float(np.max(result))
    if maximum_result > 0.0 and np.isfinite(maximum_result):
        result /= maximum_result

    return result.astype(float, copy=False)
