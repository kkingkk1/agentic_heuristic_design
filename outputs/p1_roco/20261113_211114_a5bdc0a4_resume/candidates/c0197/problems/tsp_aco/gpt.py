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

    largest = float(np.max(values)) if values.size else scale
    replacement = largest + 20.0 * scale

    clean = np.where(valid, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-9, 1e-12)

    distance_score = np.exp(
        -np.minimum(clean / max(scale, eps), 40.0)
    )
    distance_score[~valid_offdiag] = 0.0

    row_max = np.maximum(
        np.max(distance_score, axis=1, keepdims=True),
        eps,
    )
    row_strength = distance_score / row_max

    row_mass = np.maximum(
        np.sum(distance_score, axis=1, keepdims=True),
        eps,
    )
    outgoing_share = distance_score / row_mass

    incoming_mass = np.maximum(
        np.sum(distance_score, axis=0, keepdims=True),
        eps,
    )
    incoming_share = distance_score / incoming_mass

    order = np.argsort(clean, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_scale)
    rank_score[~valid_offdiag] = 0.0

    outgoing_options = np.sum(valid_offdiag, axis=1).astype(float)
    outgoing_options = np.maximum(outgoing_options, 1.0)

    future_strength = (
        np.sum(distance_score, axis=1) / outgoing_options
    )
    future_strength /= max(float(np.max(future_strength)), eps)

    destination_flexibility = future_strength[None, :]
    destination_flexibility = np.broadcast_to(
        destination_flexibility,
        (n, n),
    ).copy()
    destination_flexibility[~valid_offdiag] = 0.0

    neighborhood = min(
        n - 1,
        max(3, int(1.4 * np.sqrt(float(n))) + 1),
    )

    local_score = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = order[i]
        neighbors = neighbors[neighbors != i][:neighborhood]

        if neighbors.size == 0:
            continue

        local_values = clean[i, neighbors]
        local_scale = max(float(np.median(local_values)), eps)

        local_score[i, neighbors] = np.exp(
            -np.minimum(local_values / local_scale, 30.0)
        )

    local_score[~valid_offdiag] = 0.0

    # Keep the score directed. Incoming evidence is deliberately weaker.
    structural = (
        0.34 * row_strength
        + 0.22 * distance_score
        + 0.17 * outgoing_share
        + 0.10 * rank_score
        + 0.09 * destination_flexibility
        + 0.06 * local_score
        + 0.02 * incoming_share
    )

    structural[~valid_offdiag] = 0.0

    maximum = float(np.max(structural))
    if not np.isfinite(maximum) or maximum <= 0:
        structural = valid_offdiag.astype(float)
    else:
        structural /= maximum

    # Give every valid edge a small floor so exploration is soft rather than
    # determined by a hard candidate cutoff.
    structural = np.where(
        valid_offdiag,
        0.012 + 0.988 * structural,
        0.0,
    )

    seed_data = np.nan_to_num(
        clean,
        nan=0.0,
        posinf=replacement,
        neginf=0.0,
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
    temperature = 0.48
    counts = np.zeros((n, n), dtype=float)

    def tour_cost(tour: np.ndarray) -> float:
        return float(
            np.sum(clean[tour, np.roll(tour, -1)])
        )

    def improve_two_opt(tour: np.ndarray) -> np.ndarray:
        """Bounded asymmetric-safe 2-opt using full tour-cost checks."""

        if n < 4:
            return tour

        current_cost = tour_cost(tour)
        max_passes = 2
        max_moves = max(4, min(2 * n, 48))
        moves = 0

        for _ in range(max_passes):
            improved = False

            for a in range(n - 1):
                x = tour[a]
                y = tour[(a + 1) % n]

                # Restrict the search to promising directed successors.
                candidates = np.argsort(
                    clean[x],
                    kind="stable",
                )
                candidates = candidates[candidates != x]

                checked = 0

                for candidate in candidates:
                    matches = np.flatnonzero(tour == candidate)
                    if matches.size == 0:
                        continue

                    b = int(matches[0])

                    if b <= a + 1:
                        continue

                    if a == 0 and b == n - 1:
                        continue

                    u = tour[b]
                    v = tour[(b + 1) % n]

                    if not valid[x, u] or not valid[y, v]:
                        continue

                    proposal = tour.copy()
                    proposal[a + 1 : b + 1] = proposal[
                        a + 1 : b + 1
                    ][::-1]

                    proposal_cost = tour_cost(proposal)

                    if proposal_cost + 1e-12 < current_cost:
                        tour = proposal
                        current_cost = proposal_cost
                        improved = True
                        moves += 1
                        break

                    checked += 1
                    if checked >= min(12, n - 2):
                        break

                if improved or moves >= max_moves:
                    break

            if not improved or moves >= max_moves:
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

            if candidates.size == 0:
                break

            feasible = valid_offdiag[current, candidates]

            if np.any(feasible):
                candidates = candidates[feasible]

            edge_prior = structural[current, candidates]

            distance_term = np.exp(
                -np.minimum(
                    clean[current, candidates] / max(scale, eps),
                    40.0,
                )
            )

            future_term = future_strength[candidates]
            future_term /= max(float(np.max(future_term)), eps)

            close_term = np.exp(
                -np.minimum(
                    clean[candidates, start] / max(scale, eps),
                    40.0,
                )
            )

            # A weak rank contribution preserves local evidence without
            # eliminating globally useful but less-local edges.
            rank_term = rank_score[current, candidates]
            rank_term = np.maximum(rank_term, 0.012)

            logits = (
                1.35 * np.log(edge_prior + eps)
                + 0.92 * np.log(distance_term + eps)
                + 0.42 * np.log(future_term + eps)
                + 0.28 * np.log(close_term + eps)
                + 0.18 * np.log(rank_term + eps)
            )

            logits /= temperature
            logits -= np.max(logits)

            probabilities = np.exp(
                np.clip(logits, -60.0, 0.0)
            )

            probability_sum = float(np.sum(probabilities))

            if not np.isfinite(probability_sum) or probability_sum <= 0:
                probabilities = np.full(
                    candidates.size,
                    1.0 / candidates.size,
                    dtype=float,
                )
            else:
                probabilities /= probability_sum

            next_node = int(
                rng.choice(candidates, p=probabilities)
            )

            tour[position] = next_node
            remaining[next_node] = False
            current = next_node

        tour = improve_two_opt(tour)

        edges_from = tour
        edges_to = np.roll(tour, -1)
        counts[edges_from, edges_to] += 1.0

    counts /= float(max(samples, 1))

    result = 0.43 * structural + 0.57 * counts
    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))

    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
