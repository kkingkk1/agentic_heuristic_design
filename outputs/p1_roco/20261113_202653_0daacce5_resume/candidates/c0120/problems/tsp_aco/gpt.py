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
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    finite_values = values[np.isfinite(values)]
    largest = float(np.max(finite_values)) if finite_values.size else scale
    replacement = largest + 18.0 * scale

    clean = np.where(valid, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-9, 1e-12)
    reciprocal = 1.0 / (clean + eps)
    reciprocal[~valid_offdiag] = 0.0

    row_max = np.maximum(reciprocal.max(axis=1, keepdims=True), eps)
    col_max = np.maximum(reciprocal.max(axis=0, keepdims=True), eps)

    row_strength = reciprocal / row_max
    col_strength = reciprocal / col_max

    row_mass = np.maximum(reciprocal.sum(axis=1, keepdims=True), eps)
    col_mass = np.maximum(reciprocal.sum(axis=0, keepdims=True), eps)

    outgoing_competition = reciprocal / row_mass
    incoming_competition = reciprocal / col_mass
    exclusivity = np.sqrt(outgoing_competition * incoming_competition)

    order = np.argsort(clean, axis=1)
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-rank / max(1.0, 0.85 * np.sqrt(float(n))))
    rank_score[~valid_offdiag] = 0.0

    neighborhood = min(n - 1, max(3, int(1.25 * np.sqrt(n)) + 1))
    compatibility = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = order[i, : neighborhood + 1]
        neighbors = neighbors[neighbors != i][:neighborhood]
        if neighbors.size == 0:
            continue

        local = clean[i, neighbors]
        local_scale = max(float(np.mean(local)), eps)
        compatibility[i, neighbors] = np.exp(
            -np.minimum(local / local_scale, 30.0)
        )

    base = (
        0.36 * np.sqrt(row_strength * col_strength)
        + 0.25 * exclusivity
        + 0.21 * rank_score
        + 0.18 * compatibility
    )
    base[~valid_offdiag] = 0.0
    np.fill_diagonal(base, 0.0)

    base_max = float(np.max(base))
    if not np.isfinite(base_max) or base_max <= 0:
        base = valid_offdiag.astype(float)
    else:
        base /= base_max

    seed_data = np.nan_to_num(
        clean, nan=0.0, posinf=replacement, neginf=0.0
    )
    weights = np.arange(1, n + 1, dtype=float)
    seed_float = (
        np.sum(seed_data) * 1000003.0
        + np.sum(seed_data * weights[:, None]) * 9176.0
        + np.sum(seed_data * weights[None, :]) * 6151.0
        + n * 104729.0
    )
    seed = int(abs(seed_float)) % (2**63 - 1)
    rng = np.random.default_rng(seed)

    samples = int(np.clip(8 * n, 128, 420))
    width = int(np.clip(3.0 * np.sqrt(n), 6, 32))
    temperature = 0.36
    counts = np.zeros((n, n), dtype=float)

    def improve_two_opt(tour):
        if n < 5:
            return tour

        max_passes = 3
        for _ in range(max_passes):
            improved = False
            for a in range(n - 1):
                x = tour[a]
                y = tour[a + 1]

                for b in range(a + 2, n):
                    if b == n - 1 and a == 0:
                        continue

                    u = tour[b]
                    v = tour[(b + 1) % n]

                    old_cost = clean[x, y] + clean[u, v]
                    new_cost = clean[x, u] + clean[y, v]

                    if new_cost + 1e-12 < old_cost:
                        tour[a + 1 : b + 1] = tour[a + 1 : b + 1][::-1]
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return tour

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
                    clean[current, candidates], width - 1
                )[:width]
                candidates = candidates[nearest]

            edge_score = base[current, candidates]
            distance_term = np.exp(
                -np.minimum(clean[current, candidates] / max(scale, eps), 60.0)
            )

            remaining_after = remaining.copy()
            remaining_after[candidates] = False
            remaining_after[current] = False

            if np.any(remaining_after):
                future_count = reciprocal[candidates][:, remaining_after].sum(
                    axis=1
                )
                future_count /= max(float(np.max(future_count)), eps)
            else:
                future_count = np.ones(candidates.size, dtype=float)

            close_cost = clean[candidates, start] / max(scale, eps)
            close_term = np.exp(-np.minimum(close_cost, 60.0))

            logits = (
                1.50 * np.log(edge_score + 1e-12)
                + 1.05 * np.log(distance_term + 1e-12)
                + 0.48 * np.log(future_count + 1e-12)
                + 0.32 * np.log(close_term + 1e-12)
            )
            logits /= temperature
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            probability_sum = float(np.sum(probabilities))

            if not np.isfinite(probability_sum) or probability_sum <= 0:
                probabilities = np.full(
                    candidates.size, 1.0 / candidates.size, dtype=float
                )
            else:
                probabilities /= probability_sum

            next_node = int(rng.choice(candidates, p=probabilities))
            tour[position] = next_node
            remaining[next_node] = False
            current = next_node

        tour = improve_two_opt(tour)

        edges_from = tour
        edges_to = np.roll(tour, -1)
        counts[edges_from, edges_to] += 1.0

    counts /= float(max(samples, 1))

    result = 0.52 * base + 0.48 * counts
    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
