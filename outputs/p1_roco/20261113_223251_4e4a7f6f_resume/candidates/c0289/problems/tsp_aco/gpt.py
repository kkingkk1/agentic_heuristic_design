import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic adaptive edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    valid = finite & (raw >= 0.0)
    valid_offdiag = valid.copy()
    np.fill_diagonal(valid_offdiag, False)

    if not np.any(valid_offdiag):
        return np.zeros((n, n), dtype=float)

    values = raw[valid_offdiag]
    positive = values[values > 0.0]

    q25, q50, q75, q99 = np.quantile(
        values, [0.25, 0.50, 0.75, 0.99]
    )
    spread = float(q75 - q25)

    scale = spread if np.isfinite(spread) and spread > 0.0 else float(q50)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = max(scale * 1e-10, np.finfo(float).eps)
    replacement = max(float(q99), float(q75), scale) + 20.0 * scale

    clean = np.where(valid, raw, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid_offdiag] = 1.0 / (clean[valid_offdiag] + eps)

    row_sum = np.maximum(inverse.sum(axis=1, keepdims=True), eps)
    col_sum = np.maximum(inverse.sum(axis=0, keepdims=True), eps)

    outgoing = inverse / row_sum
    incoming = inverse / col_sum
    competition = np.sqrt(np.maximum(outgoing * incoming, 0.0))

    row_best = np.maximum(inverse.max(axis=1, keepdims=True), eps)
    col_best = np.maximum(inverse.max(axis=0, keepdims=True), eps)
    strength = np.sqrt(
        np.maximum(
            (inverse / row_best) * (inverse / col_best),
            0.0,
        )
    )

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_scale)
    rank_score[~valid_offdiag] = 0.0

    distance_score = np.exp(
        -np.minimum(clean / max(scale, eps), 60.0)
    )
    distance_score[~valid_offdiag] = 0.0

    neighborhood = min(
        n - 1,
        max(4, int(1.8 * np.sqrt(float(n)))),
    )
    local_score = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = order[i]
        candidates = candidates[candidates != i][:neighborhood]
        if candidates.size == 0:
            continue

        local_scale = max(
            float(np.median(clean[i, candidates])),
            eps,
        )
        local_score[i, candidates] = np.exp(
            -np.minimum(
                clean[i, candidates] / local_scale,
                32.0,
            )
        )

    base = (
        0.28 * strength
        + 0.24 * competition
        + 0.22 * rank_score
        + 0.16 * local_score
        + 0.10 * distance_score
    )
    base[~valid_offdiag] = 0.0

    base_max = float(np.max(base))
    if np.isfinite(base_max) and base_max > 0.0:
        base /= base_max
    else:
        base = valid_offdiag.astype(float)

    seed_data = np.nan_to_num(
        clean,
        nan=0.0,
        posinf=replacement,
        neginf=0.0,
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

    def tour_cost(tour):
        return float(np.sum(clean[tour, np.roll(tour, -1)]))

    def relocate_improve(tour):
        if n < 4:
            return tour

        current = tour_cost(tour)

        for _ in range(2):
            best_delta = 0.0
            best_move = None

            for remove_at in range(n):
                node = int(tour[remove_at])
                prev_node = int(tour[remove_at - 1])
                next_node = int(tour[(remove_at + 1) % n])

                remove_delta = (
                    clean[prev_node, next_node]
                    - clean[prev_node, node]
                    - clean[node, next_node]
                )

                for insert_at in range(n):
                    if insert_at == remove_at:
                        continue

                    anchor = int(tour[insert_at])
                    if anchor == node or anchor == prev_node:
                        continue

                    after = int(tour[(insert_at + 1) % n])
                    delta = (
                        remove_delta
                        + clean[anchor, node]
                        + clean[node, after]
                        - clean[anchor, after]
                    )

                    if delta < best_delta - 1e-12:
                        best_delta = float(delta)
                        best_move = (remove_at, insert_at)

            if best_move is None:
                break

            remove_at, insert_at = best_move
            node = int(tour[remove_at])
            reduced = np.delete(tour, remove_at)

            anchor = int(tour[insert_at])
            anchor_position = int(np.flatnonzero(reduced == anchor)[0])
            tour = np.insert(reduced, anchor_position + 1, node)
            current += best_delta

        return tour

    samples = int(np.clip(6 * n, 128, 300))
    candidate_width = int(np.clip(3.2 * np.sqrt(float(n)), 7, 30))
    counts = np.zeros((n, n), dtype=float)
    tours = []
    costs = []

    for sample_index in range(samples):
        start = (
            sample_index % n
            if sample_index < n
            else int(rng.integers(n))
        )

        tour = np.empty(n, dtype=int)
        tour[0] = start

        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        for position in range(1, n):
            candidates = np.flatnonzero(remaining)

            outgoing = valid[current, candidates]
            if np.any(outgoing):
                candidates = candidates[outgoing]

            if candidates.size > candidate_width:
                nearest = np.argpartition(
                    clean[current, candidates],
                    candidate_width - 1,
                )[:candidate_width]
                candidates = candidates[nearest]

            edge_prior = np.maximum(
                base[current, candidates],
                1e-12,
            )

            direct = np.exp(
                -np.minimum(
                    clean[current, candidates] / scale,
                    60.0,
                )
            )

            future_mask = remaining.copy()
            future_mask[current] = False
            future_mask[candidates] = False

            if np.any(future_mask):
                future = inverse[
                    candidates
                ][:, future_mask].sum(axis=1)
                future_max = float(np.max(future))
                if future_max > eps:
                    future /= future_max
                else:
                    future = np.ones(candidates.size)
            else:
                future = np.ones(candidates.size)

            closure = np.exp(
                -np.minimum(
                    clean[candidates, start] / scale,
                    60.0,
                )
            )

            logits = (
                1.70 * np.log(edge_prior)
                + 1.15 * np.log(direct + 1e-12)
                + 0.55 * np.log(future + 1e-12)
                + 0.30 * np.log(closure + 1e-12)
            )

            progress = position / max(n - 1, 1)
            temperature = 0.62 - 0.25 * progress
            logits /= max(temperature, 0.25)
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            total = float(np.sum(probabilities))

            if not np.isfinite(total) or total <= 0.0:
                probabilities = np.full(
                    candidates.size,
                    1.0 / candidates.size,
                )
            else:
                probabilities /= total

            exploration = 0.035 if position < n - 2 else 0.02
            probabilities = (
                (1.0 - exploration) * probabilities
                + exploration / candidates.size
            )

            next_node = int(
                rng.choice(candidates, p=probabilities)
            )
            tour[position] = next_node
            remaining[next_node] = False
            current = next_node

        tour = relocate_improve(tour)

        successor = np.roll(tour, -1)
        counts[tour, successor] += 1.0
        tours.append(tour)
        costs.append(tour_cost(tour))

    counts /= float(max(samples, 1))

    elite_counts = np.zeros((n, n), dtype=float)
    costs_array = np.asarray(costs, dtype=float)
    finite_costs = np.isfinite(costs_array)

    if np.any(finite_costs):
        valid_indices = np.flatnonzero(finite_costs)
        elite_size = max(
            1,
            int(np.ceil(0.18 * valid_indices.size)),
        )
        ranked = valid_indices[
            np.argsort(
                costs_array[valid_indices],
                kind="stable",
            )[:elite_size]
        ]

        for index in ranked:
            elite = tours[int(index)]
            successor = np.roll(elite, -1)
            elite_counts[elite, successor] += 1.0

        elite_counts /= float(elite_size)

    result = (
        0.43 * base
        + 0.24 * counts
        + 0.21 * elite_counts
        + 0.12 * competition
    )

    result[~valid_offdiag] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
