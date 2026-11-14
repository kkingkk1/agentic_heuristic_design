import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic edge priors for stochastic TSP solution sampling."""
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

    q25, q50, q75, q90, q99 = np.quantile(
        values, [0.25, 0.50, 0.75, 0.90, 0.99]
    )

    spread = float(q75 - q25)
    scale = spread if np.isfinite(spread) and spread > 0 else float(q50)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    upper_reference = max(float(q99), float(q75), scale)
    replacement = upper_reference + 16.0 * scale

    clean = np.where(valid, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-9, 1e-12)
    normalized_distance = np.minimum(clean / max(scale, eps), 60.0)

    reciprocal = 1.0 / (clean + eps)
    reciprocal[~valid_offdiag] = 0.0

    row_max = np.maximum(
        np.max(reciprocal, axis=1, keepdims=True), eps
    )
    col_max = np.maximum(
        np.max(reciprocal, axis=0, keepdims=True), eps
    )

    row_strength = reciprocal / row_max
    col_strength = reciprocal / col_max

    row_mass = np.maximum(
        np.sum(reciprocal, axis=1, keepdims=True), eps
    )
    col_mass = np.maximum(
        np.sum(reciprocal, axis=0, keepdims=True), eps
    )

    outgoing_competition = reciprocal / row_mass
    incoming_competition = reciprocal / col_mass
    bidirectional_competition = np.sqrt(
        outgoing_competition * incoming_competition
    )

    order = np.argsort(clean, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_scale)
    rank_score[~valid_offdiag] = 0.0

    distance_score = np.exp(-normalized_distance)
    distance_score[~valid_offdiag] = 0.0

    base = (
        0.30 * np.sqrt(row_strength * col_strength)
        + 0.25 * bidirectional_competition
        + 0.25 * rank_score
        + 0.20 * distance_score
    )
    base[~valid_offdiag] = 0.0

    base_max = float(np.max(base))
    if np.isfinite(base_max) and base_max > 0:
        base /= base_max
    else:
        base = valid_offdiag.astype(float)

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

    samples = int(np.clip(7 * n, 128, 360))
    candidate_width = int(np.clip(2.5 * np.sqrt(n), 6, 28))
    local_width = int(np.clip(2.5 * np.sqrt(n), 6, 14))
    temperature = 0.38
    exploration = 0.035

    def tour_cost(tour):
        return float(np.sum(clean[tour, np.roll(tour, -1)]))

    def targeted_two_opt(tour):
        """Exact directed 2-opt; only accept demonstrable improvements."""
        if n < 5:
            return tour

        current_cost = tour_cost(tour)

        for _ in range(3):
            improved = False

            for a in range(n - 1):
                x = int(tour[a])
                y = int(tour[a + 1])

                possible_b = np.arange(a + 2, n, dtype=int)
                if a == 0:
                    possible_b = possible_b[possible_b < n - 1]
                if possible_b.size == 0:
                    continue

                u_nodes = tour[possible_b]
                v_nodes = tour[(possible_b + 1) % n]

                boundary_hint = (
                    clean[x, u_nodes] + clean[y, v_nodes]
                )
                limit = min(local_width, possible_b.size)

                if possible_b.size > limit:
                    chosen = np.argpartition(
                        boundary_hint, limit - 1
                    )[:limit]
                    possible_b = possible_b[chosen]

                for b in possible_b:
                    candidate = tour.copy()
                    candidate[a + 1 : b + 1] = (
                        candidate[a + 1 : b + 1][::-1]
                    )
                    candidate_cost = tour_cost(candidate)

                    if candidate_cost + 1e-12 < current_cost:
                        tour = candidate
                        current_cost = candidate_cost
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return tour

    tours = []
    costs = []
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

            valid_candidates = candidates[valid[current, candidates]]
            if valid_candidates.size > 0:
                candidates = valid_candidates

            if candidates.size > candidate_width:
                nearest = np.argpartition(
                    clean[current, candidates],
                    candidate_width - 1,
                )[:candidate_width]
                candidates = candidates[nearest]

            edge_score = np.maximum(
                base[current, candidates], 1e-12
            )
            distance_term = np.exp(
                -np.minimum(
                    clean[current, candidates] / max(scale, eps),
                    60.0,
                )
            )

            remaining_after = remaining.copy()
            remaining_after[candidates] = False
            remaining_after[current] = False

            if np.any(remaining_after):
                future_count = reciprocal[
                    candidates
                ][:, remaining_after].sum(axis=1)
                future_max = float(np.max(future_count))
                if future_max > eps:
                    future_count /= future_max
                else:
                    future_count = np.ones(
                        candidates.size, dtype=float
                    )
            else:
                future_count = np.ones(
                    candidates.size, dtype=float
                )

            close_term = np.exp(
                -np.minimum(
                    clean[candidates, start] / max(scale, eps),
                    60.0,
                )
            )

            logits = (
                1.55 * np.log(edge_score)
                + 1.05 * np.log(distance_term + 1e-12)
                + 0.45 * np.log(future_count + 1e-12)
                + 0.30 * np.log(close_term + 1e-12)
            )
            logits /= temperature
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            probability_sum = float(np.sum(probabilities))

            if not np.isfinite(probability_sum) or probability_sum <= 0:
                probabilities = np.full(
                    candidates.size,
                    1.0 / candidates.size,
                    dtype=float,
                )
            else:
                probabilities /= probability_sum
                probabilities = (
                    (1.0 - exploration) * probabilities
                    + exploration / candidates.size
                )

            next_node = int(rng.choice(candidates, p=probabilities))
            tour[position] = next_node
            remaining[next_node] = False
            current = next_node

        tour = targeted_two_opt(tour)
        cost = tour_cost(tour)

        tours.append(tour)
        costs.append(cost)

        edges_from = tour
        edges_to = np.roll(tour, -1)
        counts[edges_from, edges_to] += 1.0

    counts /= float(max(samples, 1))

    costs_array = np.asarray(costs, dtype=float)
    finite_costs = np.isfinite(costs_array)

    elite_counts = np.zeros((n, n), dtype=float)
    if np.any(finite_costs):
        finite_indices = np.flatnonzero(finite_costs)
        elite_size = max(1, int(np.ceil(0.20 * finite_indices.size)))
        elite_indices = finite_indices[
            np.argsort(costs_array[finite_indices], kind="stable")[
                :elite_size
            ]
        ]

        for index in elite_indices:
            elite_tour = tours[int(index)]
            elite_counts[
                elite_tour, np.roll(elite_tour, -1)
            ] += 1.0

        elite_counts /= float(max(elite_size, 1))

    result = 0.42 * base + 0.34 * counts + 0.24 * elite_counts
    result[~valid_offdiag] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
