import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(raw) & (raw >= 0.0)
    offdiag = ~np.eye(n, dtype=bool)
    valid_offdiag = valid & offdiag

    if not np.any(valid_offdiag):
        return np.zeros((n, n), dtype=float)

    observed = raw[valid_offdiag]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed))
    penalty = maximum + 16.0 * scale
    if not np.isfinite(penalty):
        penalty = scale * 32.0

    clean = np.full((n, n), penalty, dtype=float)
    clean[valid] = raw[valid]

    reverse_valid = valid.T
    both = valid & reverse_valid
    only_forward = valid & ~reverse_valid
    only_reverse = reverse_valid & ~valid

    symmetric = np.full((n, n), penalty, dtype=float)
    symmetric[both] = 0.5 * (raw[both] + raw.T[both])
    symmetric[only_forward] = raw[only_forward]
    symmetric[only_reverse] = raw.T[only_reverse]
    clean = symmetric
    np.fill_diagonal(clean, penalty)

    eps = max(scale * 1e-9, 1e-12)
    reciprocal = 1.0 / (clean + eps)
    reciprocal[~offdiag] = 0.0
    reciprocal[clean >= penalty] = 0.0

    row_max = np.maximum(reciprocal.max(axis=1, keepdims=True), eps)
    col_max = np.maximum(reciprocal.max(axis=0, keepdims=True), eps)

    row_strength = reciprocal / row_max
    col_strength = reciprocal / col_max

    row_mass = np.maximum(reciprocal.sum(axis=1, keepdims=True), eps)
    col_mass = np.maximum(reciprocal.sum(axis=0, keepdims=True), eps)

    row_share = reciprocal / row_mass
    col_share = reciprocal / col_mass
    competition = np.sqrt(row_share * col_share)

    order = np.argsort(clean, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_decay = max(1.0, np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_decay)
    rank_score[reciprocal <= 0.0] = 0.0

    neighborhood = min(n - 1, max(3, int(np.sqrt(n)) + 1))
    local_exclusivity = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = order[i]
        candidates = candidates[candidates != i]
        candidates = candidates[reciprocal[i, candidates] > 0.0]
        candidates = candidates[:neighborhood]

        if candidates.size == 0:
            continue

        local_distances = clean[i, candidates]
        local_scale = max(float(np.median(local_distances)), eps)
        local_exclusivity[i, candidates] = np.exp(
            -local_distances / local_scale
        )

    base = (
        0.38 * np.sqrt(row_strength * col_strength)
        + 0.27 * competition
        + 0.20 * rank_score
        + 0.15 * local_exclusivity
    )
    base[reciprocal <= 0.0] = 0.0
    np.fill_diagonal(base, 0.0)

    base_max = float(np.max(base))
    if np.isfinite(base_max) and base_max > 0.0:
        base /= base_max
    else:
        base = (reciprocal > 0.0).astype(float)

    seed_values = np.nan_to_num(clean, nan=0.0, posinf=penalty, neginf=0.0)
    weights = np.arange(1, n + 1, dtype=float)
    seed_float = (
        np.sum(seed_values) * 1000003.0
        + np.sum(seed_values * weights[:, None]) * 9176.0
        + np.sum(seed_values * weights[None, :]) * 6151.0
        + n * 104729.0
    )
    seed = int(abs(seed_float)) % (2**63 - 1)
    rng = np.random.default_rng(seed)

    samples = int(np.clip(8 * n, 96, 320))
    width = int(np.clip(2.0 * np.sqrt(n) + 2.0, 4, 24))
    temperature = float(np.clip(0.34 + 0.035 * np.log1p(n), 0.34, 0.48))
    counts = np.zeros((n, n), dtype=float)

    for _ in range(samples):
        start = int(rng.integers(n))
        tour = np.empty(n, dtype=int)
        tour[0] = start

        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        for position in range(1, n):
            candidates = np.flatnonzero(remaining)

            if candidates.size > width:
                nearest = np.argpartition(
                    clean[current, candidates],
                    width - 1,
                )[:width]
                candidates = candidates[nearest]

            edge_prior = base[current, candidates]
            distance_term = np.exp(
                -np.minimum(clean[current, candidates] / scale, 60.0)
            )

            if position == n - 1:
                future_term = np.ones(candidates.size, dtype=float)
            else:
                future_mask = remaining.copy()
                future_mask[candidates] = False
                future_mask[current] = False

                if np.any(future_mask):
                    future = reciprocal[candidates][:, future_mask].sum(axis=1)
                    future_max = max(float(np.max(future)), eps)
                    future_term = future / future_max
                else:
                    future_term = np.ones(candidates.size, dtype=float)

            return_term = np.exp(
                -np.minimum(clean[candidates, start] / scale, 60.0)
            )

            logits = (
                1.25 * np.log(edge_prior + 1e-12)
                + 0.95 * np.log(distance_term + 1e-12)
                + 0.38 * np.log(future_term + 1e-12)
                + 0.25 * np.log(return_term + 1e-12)
            )
            logits /= temperature
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
            tour[position] = next_node
            remaining[next_node] = False
            current = next_node

        successors = np.roll(tour, -1)
        counts[tour, successors] += 1.0

    counts /= float(samples)
    counts = 0.5 * (counts + counts.T)
    counts[reciprocal <= 0.0] = 0.0
    np.fill_diagonal(counts, 0.0)

    result = 0.62 * base + 0.38 * counts
    result[reciprocal <= 0.0] = 0.0
    np.fill_diagonal(result, 0.0)

    result_max = float(np.max(result))
    if np.isfinite(result_max) and result_max > 0.0:
        result /= result_max

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
