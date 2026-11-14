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
        scale = float(np.mean(valid_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    replacement = float(np.max(valid_values) + 10.0 * scale)
    clean = np.where(positive, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-9, 1e-12)

    outgoing = 1.0 / (clean + eps)
    incoming = 1.0 / (clean.T + eps)

    outgoing /= np.maximum(outgoing.max(axis=1, keepdims=True), eps)
    incoming /= np.maximum(incoming.max(axis=1, keepdims=True), eps)

    base = np.sqrt(outgoing * incoming)
    np.fill_diagonal(base, 0.0)

    row_order = np.argsort(clean, axis=1)
    rank = np.empty_like(row_order, dtype=float)
    rank[np.arange(n)[:, None], row_order] = np.arange(n)
    rank_score = np.exp(-rank / max(1.0, np.sqrt(n)))
    np.fill_diagonal(rank_score, 0.0)

    compatibility = np.zeros((n, n), dtype=float)
    for i in range(n):
        nearest = row_order[i, : min(n - 1, max(3, int(np.sqrt(n))))]
        nearest = nearest[nearest != i]
        if nearest.size == 0:
            continue
        local = clean[i, nearest]
        local_scale = max(float(np.mean(local)), eps)
        compatibility[i, nearest] = np.exp(-local / local_scale)

    score = 0.50 * base + 0.30 * rank_score + 0.20 * compatibility
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    max_score = float(score.max())
    if not np.isfinite(max_score) or max_score <= 0:
        score = np.where(np.eye(n, dtype=bool), 0.0, 1.0)
    else:
        score /= max_score

    seed_value = np.nan_to_num(clean, nan=0.0, posinf=replacement)
    seed = int(
        (
            np.sum(seed_value) * 1_000_003
            + np.sum(seed_value * np.arange(1, n + 1)[:, None]) * 97
            + n * 104_729
        )
        % (2**63 - 1)
    )
    rng = np.random.default_rng(seed)

    samples = int(np.clip(8 * n, 64, 256))
    candidate_width = int(np.clip(2 * np.sqrt(n), 4, 24))
    temperature = 0.35

    counts = np.zeros((n, n), dtype=float)

    for _ in range(samples):
        start = int(rng.integers(n))
        tour = [start]
        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        while np.any(remaining):
            candidates = np.flatnonzero(remaining)
            if candidates.size > candidate_width:
                local_dist = clean[current, candidates]
                chosen = np.argpartition(
                    local_dist, candidate_width - 1
                )[:candidate_width]
                candidates = candidates[chosen]

            logits = np.log(score[current, candidates] + 1e-12)
            logits -= clean[current, candidates] / max(scale, eps)
            logits /= temperature
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            probabilities /= np.sum(probabilities)

            next_node = int(rng.choice(candidates, p=probabilities))
            counts[current, next_node] += 1.0
            tour.append(next_node)
            remaining[next_node] = False
            current = next_node

        counts[current, start] += 1.0

    counts /= max(float(samples), 1.0)
    result = 0.55 * score + 0.45 * counts

    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
