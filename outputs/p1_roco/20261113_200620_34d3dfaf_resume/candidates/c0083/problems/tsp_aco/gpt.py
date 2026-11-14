import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return soft stochastic edge priors for TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d >= 0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = d[valid]
    scale = float(np.median(values))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    replacement = float(np.max(values) + 12.0 * scale)
    clean = np.where(finite & (d >= 0), d, replacement).astype(float)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-9, 1e-12)

    distance_score = np.exp(-np.minimum(clean / scale, 60.0))
    outgoing = 1.0 / (clean + eps)
    incoming = 1.0 / (clean.T + eps)

    outgoing /= np.maximum(outgoing.max(axis=1, keepdims=True), eps)
    incoming /= np.maximum(incoming.max(axis=1, keepdims=True), eps)
    mutual_score = np.sqrt(outgoing * incoming)

    order = np.argsort(clean, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n)
    local_rank = np.exp(-ranks / max(1.0, np.sqrt(n)))

    compatibility = np.zeros((n, n), dtype=float)
    width = min(n - 1, max(3, int(np.sqrt(n))))
    for i in range(n):
        nearest = order[i, :width]
        nearest = nearest[nearest != i]
        if nearest.size:
            local_scale = max(float(np.mean(clean[i, nearest])), eps)
            compatibility[i, nearest] = np.exp(
                -np.minimum(clean[i, nearest] / local_scale, 60.0)
            )

    structural = (
        0.34 * distance_score
        + 0.30 * mutual_score
        + 0.22 * local_rank
        + 0.14 * compatibility
    )
    structural[~valid] = 0.0
    np.fill_diagonal(structural, 0.0)

    floor = 0.035
    score = floor * valid.astype(float) + (1.0 - floor) * structural
    maximum = float(score.max())
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    seed_values = np.nan_to_num(clean, nan=0.0, posinf=replacement)
    seed = int(
        (
            np.sum(seed_values) * 1_000_003
            + np.sum(seed_values * np.arange(1, n + 1)[:, None]) * 97
            + n * 104_729
        )
        % (2**63 - 1)
    )
    rng = np.random.default_rng(seed)

    samples = int(np.clip(6 * n, 64, 192))
    counts = np.zeros((n, n), dtype=float)

    for sample_index in range(samples):
        start = int(rng.integers(n))
        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        temperature = 0.28 + 0.42 * (
            sample_index / max(samples - 1, 1)
        )
        exploration = 0.10 - 0.055 * (
            sample_index / max(samples - 1, 1)
        )

        while np.any(remaining):
            candidates = np.flatnonzero(remaining & valid[current])
            if candidates.size == 0:
                candidates = np.flatnonzero(remaining)

            usage = counts[current, candidates]
            if np.any(usage > 0):
                usage = usage / max(float(usage.max()), 1.0)

            logits = (
                1.15 * np.log(score[current, candidates] + 1e-12)
                - 0.85 * clean[current, candidates] / max(scale, eps)
                + 0.16 * usage
            )
            logits /= temperature
            logits -= np.max(logits)

            soft = np.exp(np.clip(logits, -60.0, 0.0))
            soft /= max(float(np.sum(soft)), eps)

            probabilities = (
                (1.0 - exploration) * soft
                + exploration / float(candidates.size)
            )
            probabilities /= np.sum(probabilities)

            next_node = int(rng.choice(candidates, p=probabilities))
            counts[current, next_node] += 1.0
            remaining[next_node] = False
            current = next_node

        if valid[current, start]:
            counts[current, start] += 1.0

    counts /= max(float(samples), 1.0)
    result = 0.58 * score + 0.42 * counts
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
