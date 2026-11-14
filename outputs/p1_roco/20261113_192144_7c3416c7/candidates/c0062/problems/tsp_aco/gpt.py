import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d >= 0)

    if not np.any(positive):
        return np.zeros_like(d, dtype=float)

    valid_values = d[positive]
    scale = float(np.median(valid_values))
    if not np.isfinite(scale) or scale <= 0:
        positive_values = valid_values[valid_values > 0]
        if positive_values.size:
            scale = float(np.mean(positive_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    replacement = float(np.max(valid_values) + 12.0 * scale)
    clean = np.where(positive, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-9, 1e-12)

    outgoing = 1.0 / (clean + eps)
    incoming = 1.0 / (clean.T + eps)

    outgoing /= np.maximum(
        outgoing.max(axis=1, keepdims=True),
        eps,
    )
    incoming /= np.maximum(
        incoming.max(axis=1, keepdims=True),
        eps,
    )

    base = np.sqrt(outgoing * incoming)
    np.fill_diagonal(base, 0.0)

    row_order = np.argsort(clean, axis=1, kind="stable")
    rank = np.empty_like(row_order, dtype=float)
    rank[np.arange(n)[:, None], row_order] = np.arange(n)

    rank_decay = max(1.0, 0.85 * np.sqrt(n))
    rank_score = np.exp(-rank / rank_decay)
    np.fill_diagonal(rank_score, 0.0)

    compatibility = np.zeros((n, n), dtype=float)
    neighborhood = int(np.clip(2.5 * np.sqrt(n), 4, 12))

    for i in range(n):
        nearest = row_order[i, : min(n - 1, neighborhood)]
        nearest = nearest[nearest != i]

        if nearest.size == 0:
            continue

        local = clean[i, nearest]
        local_scale = max(float(np.median(local)), eps)
        compatibility[i, nearest] = np.exp(-local / local_scale)

    score = (
        0.56 * base
        + 0.29 * rank_score
        + 0.15 * compatibility
    )
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    max_score = float(score.max())
    if not np.isfinite(max_score) or max_score <= 0:
        score = np.where(np.eye(n, dtype=bool), 0.0, 1.0)
    else:
        score /= max_score

    seed_value = np.nan_to_num(
        clean,
        nan=0.0,
        posinf=replacement,
        neginf=replacement,
    )
    row_weights = np.arange(1, n + 1, dtype=float)[:, None]
    seed = int(
        (
            np.sum(seed_value) * 1_000_003
            + np.sum(seed_value * row_weights) * 97
            + np.sum(seed_value.T * row_weights) * 53
            + n * 104_729
        )
        % (2**63 - 1)
    )
    rng = np.random.default_rng(seed)

    samples = int(np.clip(12 * n, 96, 384))
    candidate_width = int(np.clip(1.6 * np.sqrt(n) + 2, 4, 18))
    temperature = float(np.clip(0.28 + 0.04 * np.log1p(n), 0.28, 0.48))

    counts = np.zeros((n, n), dtype=float)

    for _ in range(samples):
        start = int(rng.integers(n))
        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        while np.any(remaining):
            candidates = np.flatnonzero(remaining)

            if candidates.size > candidate_width:
                local_dist = clean[current, candidates]
                chosen = np.argpartition(
                    local_dist,
                    candidate_width - 1,
                )[:candidate_width]
                candidates = candidates[chosen]

            logits = np.log(score[current, candidates] + 1e-12)
            logits -= clean[current, candidates] / max(scale, eps)
            logits /= temperature
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            probability_sum = float(probabilities.sum())

            if not np.isfinite(probability_sum) or probability_sum <= 0:
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

    result = 0.58 * score + 0.42 * counts
    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
