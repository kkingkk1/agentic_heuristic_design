import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge priors for stochastic TSP solution sampling."""
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
    positive = values[values > 0]

    scale = float(np.median(values))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    largest = float(np.max(values)) if values.size else scale
    replacement = largest + 24.0 * scale

    clean = np.where(valid, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-10, 1e-12)
    inverse = 1.0 / (clean + eps)
    inverse[~valid_offdiag] = 0.0

    row_sum = np.maximum(inverse.sum(axis=1, keepdims=True), eps)
    col_sum = np.maximum(inverse.sum(axis=0, keepdims=True), eps)

    outgoing = inverse / row_sum
    incoming = inverse / col_sum
    competition = np.sqrt(outgoing * incoming)

    row_best = np.maximum(inverse.max(axis=1, keepdims=True), eps)
    col_best = np.maximum(inverse.max(axis=0, keepdims=True), eps)
    strength = np.sqrt(inverse / row_best * inverse / col_best)

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_decay = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_decay)
    rank_score[~valid_offdiag] = 0.0

    neighborhood = min(n - 1, max(4, int(1.5 * np.sqrt(n))))
    compatibility = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = order[i, : neighborhood + 1]
        neighbors = neighbors[neighbors != i][:neighborhood]
        if neighbors.size:
            local = clean[i, neighbors]
            local_scale = max(float(np.mean(local)), eps)
            compatibility[i, neighbors] = np.exp(
                -np.minimum(local / local_scale, 32.0)
            )

    base = (
        0.34 * strength
        + 0.25 * competition
        + 0.23 * rank_score
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
    seed_value = (
        np.sum(seed_data) * 1000003.0
        + np.sum(seed_data * weights[:, None]) * 9176.0
        + np.sum(seed_data * weights[None, :]) * 6151.0
        + n * 104729.0
    )
    seed = int(abs(seed_value)) % (2**63 - 1)
    rng = np.random.default_rng(seed)

    symmetric = bool(np.allclose(clean, clean.T, rtol=1e-10, atol=1e-12))

    def tour_cost(tour):
        return float(np.sum(clean[tour, np.roll(tour, -1)]))

    def improve_symmetric(tour):
        if n < 4:
            return tour

        for _ in range(4):
            changed = False
            for i in range(n - 1):
                a = tour[i]
                b = tour[(i + 1) % n]

                max_j = n - 1 if i > 0 else n - 2
                for j in range(i + 2, max_j + 1):
                    c = tour[j]
                    e = tour[(j + 1) % n]

                    old = clean[a, b] + clean[c, e]
                    new = clean[a, c] + clean[b, e]

                    if new + 1e-12 < old:
                        tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                        changed = True
                        break

                if changed:
                    break

            if not changed:
                break

        return tour

    def improve_asymmetric(tour):
        if n < 4:
            return tour

        current_cost = tour_cost(tour)

        for _ in range(2):
            changed = False

            for i in range(n):
                candidate_node = tour[i]
                candidate_tour = np.delete(tour, i)

                for j in range(n - 1):
                    trial = np.insert(candidate_tour, j + 1, candidate_node)
                    trial_cost = tour_cost(trial)

                    if trial_cost + 1e-12 < current_cost:
                        tour = trial
                        current_cost = trial_cost
                        changed = True
                        break

                if changed:
                    break

            if not changed:
                break

        return tour

    def improve(tour):
        return improve_symmetric(tour) if symmetric else improve_asymmetric(tour)

    def nearest_neighbor(start):
        tour = np.empty(n, dtype=int)
        tour[0] = start

        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        for pos in range(1, n):
            candidates = np.flatnonzero(remaining)
            next_node = candidates[np.argmin(clean[current, candidates])]
            tour[pos] = next_node
            remaining[next_node] = False
            current = next_node

        return improve(tour)

    counts = np.zeros((n, n), dtype=float)
    elite_counts = np.zeros((n, n), dtype=float)

    elite_tours = []
    elite_costs = []

    deterministic_starts = np.unique(
        np.concatenate(
            [
                np.arange(n),
                np.argsort(clean.sum(axis=1))[: min(n, 8)],
                np.argsort(clean.min(axis=1))[: min(n, 8)],
            ]
        )
    )

    for start in deterministic_starts:
        tour = nearest_neighbor(int(start))
        cost = tour_cost(tour)
        elite_tours.append(tour)
        elite_costs.append(cost)

        elite_counts[tour, np.roll(tour, -1)] += 1.0

    sample_count = int(np.clip(10 * n, 160, 480))
    candidate_width = int(np.clip(3.5 * np.sqrt(n), 7, 36))

    for sample_index in range(sample_count):
        start = int(rng.integers(n))
        tour = np.empty(n, dtype=int)
        tour[0] = start

        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        for position in range(1, n):
            candidates = np.flatnonzero(remaining)

            if candidates.size > candidate_width:
                local_distances = clean[current, candidates]
                keep = np.argpartition(
                    local_distances, candidate_width - 1
                )[:candidate_width]
                candidates = candidates[keep]

            edge_prior = np.maximum(base[current, candidates], 1e-12)
            distance_score = np.exp(
                -np.minimum(clean[current, candidates] / scale, 60.0)
            )

            future_remaining = remaining.copy()
            future_remaining[current] = False
            future_remaining[candidates] = False

            if np.any(future_remaining):
                future_score = inverse[candidates][:, future_remaining].sum(
                    axis=1
                )
                future_score /= max(float(np.max(future_score)), eps)
            else:
                future_score = np.ones(candidates.size, dtype=float)

            closure_score = np.exp(
                -np.minimum(clean[candidates, start] / scale, 60.0)
            )

            if candidates.size > 1:
                second_best = np.partition(
                    clean[candidates], 1, axis=1
                )[:, 1]
                regret = np.maximum(
                    second_best - clean[candidates, start], 0.0
                )
                regret /= max(float(np.max(regret)), scale)
            else:
                regret = np.ones(1, dtype=float)

            logits = (
                1.65 * np.log(edge_prior)
                + 1.10 * np.log(distance_score + 1e-12)
                + 0.55 * np.log(future_score + 1e-12)
                + 0.38 * np.log(closure_score + 1e-12)
                + 0.18 * regret
            )

            progress = position / max(n - 1, 1)
            temperature = 0.52 - 0.20 * progress
            logits /= max(temperature, 0.22)
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            total = float(np.sum(probabilities))

            if not np.isfinite(total) or total <= 0:
                probabilities = np.full(
                    candidates.size, 1.0 / candidates.size
                )
            else:
                probabilities /= total

            next_node = int(rng.choice(candidates, p=probabilities))
            tour[position] = next_node
            remaining[next_node] = False
            current = next_node

        tour = improve(tour)
        counts[tour, np.roll(tour, -1)] += 1.0

        cost = tour_cost(tour)
        if len(elite_tours) < 12:
            elite_tours.append(tour.copy())
            elite_costs.append(cost)
        else:
            worst = int(np.argmax(elite_costs))
            if cost < elite_costs[worst]:
                elite_tours[worst] = tour.copy()
                elite_costs[worst] = cost

    counts /= float(max(sample_count, 1))

    if elite_tours:
        order_elite = np.argsort(np.asarray(elite_costs))
        top_k = order_elite[: min(12, len(order_elite))]
        for rank, index in enumerate(top_k):
            weight = 1.0 / (1.0 + 0.35 * rank)
            tour = elite_tours[int(index)]
            elite_counts[tour, np.roll(tour, -1)] += weight

        elite_counts /= max(float(np.max(elite_counts)), 1.0)

    result = 0.46 * base + 0.40 * counts + 0.14 * elite_counts
    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
