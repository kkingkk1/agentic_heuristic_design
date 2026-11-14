import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a reproducible symmetric edge-priority matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_offdiag = np.isfinite(d) & ~np.eye(n, dtype=bool)
    valid_values = d[finite_offdiag]
    valid_values = valid_values[valid_values >= 0.0]

    if valid_values.size == 0:
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    positive_values = valid_values[valid_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else float(np.median(valid_values))
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    replacement = float(
        np.max(valid_values) + 10.0 * scale
    )
    if not np.isfinite(replacement):
        replacement = 100.0 * scale

    clean = np.where(
        np.isfinite(d) & (d >= 0.0),
        d,
        replacement,
    ).astype(float, copy=True)

    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, replacement)

    eps = np.finfo(float).eps * max(scale, 1.0)

    inverse = 1.0 / (clean + 0.08 * scale + eps)
    inverse /= np.maximum(
        np.max(inverse, axis=1, keepdims=True),
        eps,
    )

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))
    np.fill_diagonal(rank_score, 0.0)

    neighborhood_width = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    local_scales = np.empty(n, dtype=float)
    for i in range(n):
        neighbors = order[i, :neighborhood_width]
        local_scales[i] = float(np.mean(clean[i, neighbors]))
    local_scales = np.maximum(local_scales, 0.05 * scale)

    local_score = np.exp(
        -clean / np.maximum(local_scales[:, None], eps)
    )
    local_score /= np.maximum(
        np.max(local_score, axis=1, keepdims=True),
        eps,
    )
    np.fill_diagonal(local_score, 0.0)

    candidate_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        candidates = order[i, :neighborhood_width]
        candidate_mask[i, candidates] = True
    mutual_bonus = (candidate_mask & candidate_mask.T).astype(float)
    np.fill_diagonal(mutual_bonus, 0.0)

    relative_score = np.sqrt(
        np.maximum(inverse, 0.0) * np.maximum(inverse.T, 0.0)
    )

    base = (
        0.42 * relative_score
        + 0.25 * rank_score
        + 0.20 * local_score
        + 0.13 * mutual_bonus
    )
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)
    base[~np.isfinite(base)] = 0.0

    maximum = float(np.max(base))
    if maximum > 0.0:
        base /= maximum

    seed_values = np.nan_to_num(clean, nan=0.0, posinf=replacement)
    weights = np.arange(1, n + 1, dtype=float)
    seed_float = (
        np.sum(seed_values) * 1_000_003.0
        + np.sum(seed_values * weights[:, None]) * 97.0
        + np.sum(seed_values * weights[None, :]) * 53.0
        + n * 104_729.0
    )
    seed = int(abs(seed_float)) % (2**63 - 1)
    rng = np.random.default_rng(seed)

    samples = int(np.clip(6 * n, 48, 192))
    candidate_width = int(np.clip(2 * np.sqrt(n) + 2, 4, 20))
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

            logits = (
                np.log(base[current, candidates] + 1e-12)
                - clean[current, candidates] / max(scale, eps)
            )
            logits /= 0.55
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            total = float(np.sum(probabilities))

            if not np.isfinite(total) or total <= 0.0:
                probabilities = np.full(
                    candidates.size,
                    1.0 / candidates.size,
                    dtype=float,
                )
            else:
                probabilities /= total

            next_node = int(rng.choice(candidates, p=probabilities))
            counts[current, next_node] += 1.0
            remaining[next_node] = False
            current = next_node

        counts[current, start] += 1.0

    counts /= float(max(samples, 1))
    counts = 0.5 * (counts + counts.T)

    result = 0.62 * base + 0.38 * counts
    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if maximum > 0.0 and np.isfinite(maximum):
        result /= maximum

    return np.clip(result, 0.0, 1.0).astype(float, copy=False)
