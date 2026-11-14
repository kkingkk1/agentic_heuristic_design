import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge priors for stochastic TSP solution sampling."""
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
    valid_edge = valid.copy()
    np.fill_diagonal(valid_edge, False)

    if not np.any(valid_edge):
        return np.zeros((n, n), dtype=float)

    edge_values = d[valid_edge]
    positive = edge_values[edge_values > 0]
    scale = float(np.median(edge_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    largest = float(np.max(edge_values))
    replacement = largest + 20.0 * scale
    clean = np.where(valid, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-9, 1e-12)
    reciprocal = 1.0 / (clean + eps)
    reciprocal[~valid_edge] = 0.0

    row_mass = np.maximum(reciprocal.sum(axis=1, keepdims=True), eps)
    col_mass = np.maximum(reciprocal.sum(axis=0, keepdims=True), eps)
    outgoing = reciprocal / row_mass
    incoming = reciprocal / col_mass
    exclusivity = np.sqrt(outgoing * incoming)

    row_max = np.maximum(reciprocal.max(axis=1, keepdims=True), eps)
    col_max = np.maximum(reciprocal.max(axis=0, keepdims=True), eps)
    local_strength = np.sqrt(
        (reciprocal / row_max) * (reciprocal / col_max)
    )

    order = np.argsort(clean, axis=1)
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_scale)
    rank_score[~valid_edge] = 0.0

    neighborhood = min(n - 1, max(3, int(1.25 * np.sqrt(n)) + 1))
    compatibility = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = order[i, : neighborhood + 1]
        neighbors = neighbors[neighbors != i][:neighborhood]
        if neighbors.size:
            local_scale = max(float(np.mean(clean[i, neighbors])), eps)
            compatibility[i, neighbors] = np.exp(
                -np.minimum(clean[i, neighbors] / local_scale, 30.0)
            )

    base = (
        0.34 * local_strength
        + 0.27 * exclusivity
        + 0.23 * rank_score
        + 0.16 * compatibility
    )
    base[~valid_edge] = 0.0

    def robust_normalize(matrix: np.ndarray) -> np.ndarray:
        values = matrix[valid_edge]
        positive_values = values[values > 0]
        if positive_values.size == 0:
            return np.zeros_like(matrix)
        reference = float(np.quantile(positive_values, 0.95))
        if not np.isfinite(reference) or reference <= 0:
            reference = float(np.max(positive_values))
        if not np.isfinite(reference) or reference <= 0:
            return np.zeros_like(matrix)
        normalized = np.clip(matrix / reference, 0.0, 1.0)
        normalized[~valid_edge] = 0.0
        return normalized

    base = robust_normalize(base)

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
    temperature = 0.40

    tour_counts = np.zeros((n, n), dtype=float)
    two_opt_gain = np.zeros((n, n), dtype=float)

    def improve_two_opt(tour: np.ndarray) -> np.ndarray:
        if n < 5:
            return tour

        for _ in range(3):
            changed = False

            for a in range(n - 1):
                x = int(tour[a])
                y = int(tour[a + 1])

                for b in range(a + 2, n):
                    if a == 0 and b == n - 1:
                        continue

                    u = int(tour[b])
                    v = int(tour[(b + 1) % n])

                    old_cost = clean[x, y] + clean[u, v]
                    new_cost = clean[x, u] + clean[y, v]
                    gain = old_cost - new_cost

                    if gain > 1e-12 and np.isfinite(gain):
                        tour[a + 1 : b + 1] = tour[a + 1 : b + 1][::-1]

                        gain_weight = min(
                            float(gain / max(scale, eps)), 8.0
                        )
                        if valid_edge[x, u]:
                            two_opt_gain[x, u] += gain_weight
                        if valid_edge[y, v]:
                            two_opt_gain[y, v] += gain_weight

                        changed = True
                        break

                if changed:
                    break

            if not changed:
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

            valid_candidates = candidates[valid_edge[current, candidates]]
            if valid_candidates.size:
                candidates = valid_candidates

            if candidates.size > width:
                nearest = np.argpartition(
                    clean[current, candidates], width - 1
                )[:width]
                candidates = candidates[nearest]

            edge_score = base[current, candidates]
            distance_term = np.exp(
                -np.minimum(clean[current, candidates] / scale, 60.0)
            )

            remaining_after = remaining.copy()
            remaining_after[candidates] = False
            remaining_after[current] = False

            if np.any(remaining_after):
                future_count = reciprocal[candidates][:, remaining_after].sum(
                    axis=1
                )
                max_future = float(np.max(future_count))
                if max_future > eps:
                    future_count /= max_future
                else:
                    future_count = np.ones(candidates.size, dtype=float)
            else:
                future_count = np.ones(candidates.size, dtype=float)

            close_term = np.exp(
                -np.minimum(clean[candidates, start] / scale, 60.0)
            )

            logits = (
                1.45 * np.log(edge_score + 1e-12)
                + 1.00 * np.log(distance_term + 1e-12)
                + 0.50 * np.log(future_count + 1e-12)
                + 0.30 * np.log(close_term + 1e-12)
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

        source = tour
        target = np.roll(tour, -1)
        tour_counts[source, target] += 1.0

    tour_counts /= float(max(samples, 1))
    tour_counts = robust_normalize(tour_counts)
    two_opt_gain = robust_normalize(two_opt_gain)

    result = (
        0.50 * base
        + 0.34 * tour_counts
        + 0.16 * two_opt_gain
    )

    valid_values = result[valid_edge]
    if valid_values.size:
        floor = float(np.quantile(valid_values, 0.10)) * 0.04
        if not np.isfinite(floor) or floor <= 0:
            floor = 1e-6
        result[valid_edge] = np.maximum(result[valid_edge], floor)

    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
