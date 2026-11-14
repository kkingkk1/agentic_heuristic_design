import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic multi-view edge priors for stochastic TSP sampling."""
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
    upper = max(float(q99), float(q75), scale)
    replacement = upper + 16.0 * scale

    clean = np.where(valid, raw, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid_offdiag] = 1.0 / (
        clean[valid_offdiag] + eps
    )

    row_sum = np.maximum(
        inverse.sum(axis=1, keepdims=True), eps
    )
    col_sum = np.maximum(
        inverse.sum(axis=0, keepdims=True), eps
    )

    outgoing = inverse / row_sum
    incoming = inverse / col_sum
    competition = np.sqrt(
        np.maximum(outgoing * incoming, 0.0)
    )

    row_best = np.maximum(
        inverse.max(axis=1, keepdims=True), eps
    )
    col_best = np.maximum(
        inverse.max(axis=0, keepdims=True), eps
    )
    strength = np.sqrt(
        np.maximum(
            inverse / row_best * inverse / col_best,
            0.0,
        )
    )

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(
        n, dtype=float
    )

    rank_scale = max(1.0, 0.95 * np.sqrt(float(n)))
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
    compatibility = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = order[i]
        candidates = candidates[candidates != i]
        candidates = candidates[:neighborhood]

        if candidates.size:
            local_scale = max(
                float(np.median(clean[i, candidates])),
                eps,
            )
            compatibility[i, candidates] = np.exp(
                -np.minimum(
                    clean[i, candidates] / local_scale,
                    32.0,
                )
            )

    base = (
        0.28 * strength
        + 0.24 * competition
        + 0.26 * rank_score
        + 0.22 * compatibility
    )
    base[~valid_offdiag] = 0.0

    base_max = float(np.max(base))
    if np.isfinite(base_max) and base_max > 0.0:
        base /= base_max
    else:
        base = valid_offdiag.astype(float)

    undirected_valid = valid_offdiag & valid_offdiag.T
    mst_stability = np.zeros((n, n), dtype=float)

    if np.any(undirected_valid):
        upper_i, upper_j = np.triu_indices(n, k=1)
        keep = undirected_valid[upper_i, upper_j]
        edge_i = upper_i[keep]
        edge_j = upper_j[keep]
        edge_cost = 0.5 * (
            clean[edge_i, edge_j]
            + clean[edge_j, edge_i]
        )

        edge_count = edge_i.size
        trials = int(np.clip(
            14 + 2.5 * np.sqrt(float(n)),
            16,
            36,
        ))
        counts_mst = np.zeros(edge_count, dtype=float)

        pair_code = (
            (edge_i.astype(np.int64) + 1) * 73856093
            ^ (edge_j.astype(np.int64) + 1) * 19349663
            ^ (n + 1) * 83492791
        ).astype(float)

        def find(parent, node):
            root = node
            while parent[root] != root:
                root = parent[root]

            while parent[node] != node:
                next_node = parent[node]
                parent[node] = root
                node = next_node

            return root

        for trial in range(trials):
            phase = (
                pair_code * 1e-6
                + (trial + 1) * 1.61803398875
            )
            perturbed = edge_cost * (
                1.0 + 0.18 * np.sin(phase)
            )
            edge_order = np.lexsort(
                (edge_j, edge_i, perturbed)
            )

            parent = np.arange(n, dtype=int)
            rank = np.zeros(n, dtype=np.int8)
            selected = 0

            for index in edge_order:
                left = int(edge_i[index])
                right = int(edge_j[index])
                root_left = find(parent, left)
                root_right = find(parent, right)

                if root_left == root_right:
                    continue

                if rank[root_left] < rank[root_right]:
                    parent[root_left] = root_right
                elif rank[root_left] > rank[root_right]:
                    parent[root_right] = root_left
                else:
                    parent[root_right] = root_left
                    rank[root_left] += 1

                counts_mst[index] += 1.0
                selected += 1

                if selected == n - 1:
                    break

        stability = counts_mst / float(max(trials, 1))
        mst_stability[edge_i, edge_j] = stability
        mst_stability[edge_j, edge_i] = stability

    stability_max = float(np.max(mst_stability))
    if np.isfinite(stability_max) and stability_max > 0.0:
        mst_stability /= stability_max

    seed_data = np.nan_to_num(
        clean,
        nan=0.0,
        posinf=replacement,
        neginf=0.0,
    )
    node_weights = np.arange(1, n + 1, dtype=float)
    seed_value = (
        np.sum(seed_data) * 1000003.0
        + np.sum(seed_data * node_weights[:, None]) * 9176.0
        + np.sum(seed_data * node_weights[None, :]) * 6151.0
        + n * 104729.0
    )
    seed = int(abs(seed_value)) % (2**63 - 1)
    rng = np.random.default_rng(seed)

    def tour_cost(tour):
        return float(
            np.sum(clean[tour, np.roll(tour, -1)])
        )

    def improve_asymmetric(tour):
        if n < 4:
            return tour

        current = tour_cost(tour)

        for _ in range(3):
            changed = False

            for remove_at in range(n):
                node = int(tour[remove_at])
                reduced = np.delete(tour, remove_at)

                for insert_after in range(n - 1):
                    candidate = np.insert(
                        reduced,
                        insert_after + 1,
                        node,
                    )
                    candidate_cost = tour_cost(candidate)

                    if candidate_cost + 1e-12 < current:
                        tour = candidate
                        current = candidate_cost
                        changed = True
                        break

                if changed:
                    break

            if not changed:
                break

        return tour

    samples = int(np.clip(10 * n, 160, 420))
    candidate_width = int(np.clip(
        3.6 * np.sqrt(float(n)),
        7,
        32,
    ))

    counts = np.zeros((n, n), dtype=float)
    tours = []
    costs = []

    exploration = 0.055

    for _ in range(samples):
        start = int(rng.integers(n))
        tour = np.empty(n, dtype=int)
        tour[0] = start

        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        for position in range(1, n):
            candidates = np.flatnonzero(remaining)

            outgoing_valid = valid[current, candidates]
            if np.any(outgoing_valid):
                candidates = candidates[outgoing_valid]

            if candidates.size > candidate_width:
                nearest = np.argpartition(
                    clean[current, candidates],
                    candidate_width - 1,
                )[:candidate_width]
                candidates = candidates[nearest]

            edge_prior = np.maximum(
                base[current, candidates]
                + 0.24 * mst_stability[
                    current, candidates
                ],
                1e-12,
            )

            direct_term = np.exp(
                -np.minimum(
                    clean[current, candidates] / scale,
                    60.0,
                )
            )

            future_remaining = remaining.copy()
            future_remaining[current] = False
            future_remaining[candidates] = False

            if np.any(future_remaining):
                future = inverse[
                    candidates
                ][:, future_remaining].sum(axis=1)
                future_max = float(np.max(future))
                if future_max > eps:
                    future /= future_max
                else:
                    future = np.ones(
                        candidates.size,
                        dtype=float,
                    )
            else:
                future = np.ones(
                    candidates.size,
                    dtype=float,
                )

            closure = np.exp(
                -np.minimum(
                    clean[candidates, start] / scale,
                    60.0,
                )
            )

            logits = (
                1.45 * np.log(edge_prior)
                + 1.00 * np.log(direct_term + 1e-12)
                + 0.58 * np.log(future + 1e-12)
                + 0.40 * np.log(closure + 1e-12)
            )

            progress = position / max(n - 1, 1)
            temperature = 0.62 - 0.24 * progress
            logits /= max(temperature, 0.28)
            logits -= np.max(logits)

            probabilities = np.exp(
                np.clip(logits, -60.0, 0.0)
            )
            total = float(np.sum(probabilities))

            if not np.isfinite(total) or total <= 0.0:
                probabilities = np.full(
                    candidates.size,
                    1.0 / candidates.size,
                )
            else:
                probabilities /= total
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

        tour = improve_asymmetric(tour)
        counts[
            tour,
            np.roll(tour, -1),
        ] += 1.0
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
        ordered_indices = valid_indices[
            np.argsort(
                costs_array[valid_indices],
                kind="stable",
            )[:elite_size]
        ]

        for index in ordered_indices:
            elite = tours[int(index)]
            elite_counts[
                elite,
                np.roll(elite, -1),
            ] += 1.0

        elite_counts /= float(max(elite_size, 1))

    result = (
        0.38 * base
        + 0.22 * mst_stability
        + 0.25 * counts
        + 0.15 * elite_counts
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
