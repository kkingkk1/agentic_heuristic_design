import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic edge priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d >= 0.0)
    valid_edge = valid.copy()
    np.fill_diagonal(valid_edge, False)

    if not np.any(valid_edge):
        return np.zeros((n, n), dtype=float)

    observed = d[valid_edge]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    largest = float(np.max(observed)) if observed.size else scale
    if not np.isfinite(largest):
        largest = scale

    replacement = largest + 24.0 * scale
    clean = np.where(valid, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-10, 1e-12)
    closeness = 1.0 / (clean + eps)
    closeness[~valid_edge] = 0.0

    row_max = np.maximum(np.max(closeness, axis=1, keepdims=True), eps)
    col_max = np.maximum(np.max(closeness, axis=0, keepdims=True), eps)
    row_peak = closeness / row_max
    col_peak = closeness / col_max

    row_mass = np.maximum(np.sum(closeness, axis=1, keepdims=True), eps)
    col_mass = np.maximum(np.sum(closeness, axis=0, keepdims=True), eps)
    outgoing_share = closeness / row_mass
    incoming_share = closeness / col_mass
    exclusivity = np.sqrt(outgoing_share * incoming_share)

    order = np.argsort(clean, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.72 * np.sqrt(float(n)))
    rank_prior = np.exp(-rank / rank_scale)
    rank_prior[~valid_edge] = 0.0

    neighborhood_size = min(n - 1, max(3, int(1.5 * np.sqrt(float(n)))))
    compatibility = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = order[i]
        neighbors = neighbors[neighbors != i][:neighborhood_size]
        if neighbors.size == 0:
            continue

        local_values = clean[i, neighbors]
        local_mean = max(float(np.mean(local_values)), eps)
        compatibility[i, neighbors] = np.exp(
            -np.minimum(local_values / local_mean, 32.0)
        )

    reciprocal_cost = np.minimum(scale / (clean + eps), 1.0)
    reciprocal_cost[~valid_edge] = 0.0

    base = (
        0.28 * np.sqrt(row_peak * col_peak)
        + 0.22 * exclusivity
        + 0.19 * rank_prior
        + 0.16 * compatibility
        + 0.15 * reciprocal_cost
    )
    base[~valid_edge] = 0.0
    np.fill_diagonal(base, 0.0)

    base_max = float(np.max(base))
    if not np.isfinite(base_max) or base_max <= 0.0:
        base = valid_edge.astype(float)
    else:
        base /= base_max

    seed_matrix = np.nan_to_num(
        clean,
        nan=0.0,
        posinf=replacement,
        neginf=0.0,
    )
    row_weights = np.arange(1.0, n + 1.0)
    col_weights = np.arange(3.0, n + 3.0)

    seed_value = (
        np.sum(seed_matrix) * 1000003.0
        + np.sum(seed_matrix * row_weights[:, None]) * 9176.0
        + np.sum(seed_matrix * col_weights[None, :]) * 6151.0
        + n * 104729.0
    )
    seed = int(abs(float(seed_value))) % (2**63 - 1)
    rng = np.random.default_rng(seed)

    samples = int(np.clip(7 * n, 144, 384))
    candidate_width = int(np.clip(2.5 * np.sqrt(float(n)) + 4.0, 7, 28))
    temperature = 0.42
    counts = np.zeros((n, n), dtype=float)

    def improve_relocate(tour: np.ndarray) -> np.ndarray:
        """Exact first-improvement relocate search for directed TSP tours."""
        if n < 4:
            return tour

        for _ in range(3):
            improved = False

            for source_pos in range(1, n):
                source = int(tour[source_pos])
                predecessor = int(tour[source_pos - 1])
                successor = int(tour[(source_pos + 1) % n])

                removal_cost = (
                    clean[predecessor, source]
                    + clean[source, successor]
                )

                for insert_pos in range(n):
                    if insert_pos == source_pos:
                        continue
                    if insert_pos == source_pos - 1:
                        continue

                    insert_node = int(tour[insert_pos])
                    insert_successor = int(tour[(insert_pos + 1) % n])

                    old_cost = removal_cost + clean[insert_node, insert_successor]
                    new_cost = (
                        clean[predecessor, successor]
                        + clean[insert_node, source]
                        + clean[source, insert_successor]
                    )

                    if new_cost + 1e-12 < old_cost:
                        moved = int(tour[source_pos])

                        if source_pos < insert_pos:
                            tour[source_pos:insert_pos] = tour[
                                source_pos + 1 : insert_pos + 1
                            ]
                            tour[insert_pos] = moved
                        else:
                            tour[insert_pos + 2 : source_pos + 1] = tour[
                                insert_pos + 1 : source_pos
                            ]
                            tour[insert_pos + 1] = moved

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
            all_candidates = np.flatnonzero(remaining)

            if all_candidates.size > candidate_width:
                nearest_count = max(2, candidate_width // 2)
                nearest_indices = np.argpartition(
                    clean[current, all_candidates],
                    nearest_count - 1,
                )[:nearest_count]
                nearest = all_candidates[nearest_indices]

                prior_count = candidate_width - nearest.size
                if prior_count > 0:
                    prior_values = base[current, all_candidates].copy()
                    prior_values[np.isin(all_candidates, nearest)] = -1.0
                    prior_indices = np.argpartition(
                        prior_values,
                        -prior_count,
                    )[-prior_count:]
                    prior_candidates = all_candidates[prior_indices]
                    candidates = np.unique(
                        np.concatenate((nearest, prior_candidates))
                    )
                else:
                    candidates = nearest
            else:
                candidates = all_candidates

            if candidates.size == 0:
                break

            remaining_after = remaining.copy()
            remaining_after[candidates] = False

            edge_prior = np.maximum(base[current, candidates], 1e-12)
            distance_score = np.exp(
                -np.minimum(clean[current, candidates] / scale, 60.0)
            )

            if np.any(remaining_after):
                future_nodes = np.flatnonzero(remaining_after)
                future_cost = np.min(
                    clean[candidates[:, None], future_nodes[None, :]],
                    axis=1,
                )
                future_score = np.exp(
                    -np.minimum(future_cost / scale, 60.0)
                )

                future_coverage = closeness[candidates][:, future_nodes].sum(
                    axis=1
                )
                coverage_max = max(float(np.max(future_coverage)), eps)
                future_coverage = future_coverage / coverage_max
            else:
                future_score = np.ones(candidates.size, dtype=float)
                future_coverage = np.ones(candidates.size, dtype=float)

            close_score = np.exp(
                -np.minimum(clean[candidates, start] / scale, 60.0)
            )

            logits = (
                1.55 * np.log(edge_prior)
                + 0.92 * np.log(distance_score + 1e-12)
                + 0.62 * np.log(future_score + 1e-12)
                + 0.28 * np.log(future_coverage + 1e-12)
                + 0.38 * np.log(close_score + 1e-12)
            )

            logits /= temperature
            logits -= np.max(logits)
            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            total = float(np.sum(probabilities))

            if not np.isfinite(total) or total <= 0.0:
                probabilities.fill(1.0 / probabilities.size)
            else:
                probabilities /= total

            next_node = int(rng.choice(candidates, p=probabilities))
            tour[position] = next_node
            remaining[next_node] = False
            current = next_node

        if np.any(remaining):
            missing = np.flatnonzero(remaining)
            tour[position:] = missing[: n - position]

        tour = improve_relocate(tour)

        source_nodes = tour
        target_nodes = np.roll(tour, -1)
        counts[source_nodes, target_nodes] += 1.0

    counts /= float(max(samples, 1))

    result = 0.56 * base + 0.44 * counts
    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return result.astype(float, copy=False)
