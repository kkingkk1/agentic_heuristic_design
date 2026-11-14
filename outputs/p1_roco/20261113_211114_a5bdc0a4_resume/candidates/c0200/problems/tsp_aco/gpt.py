import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.maximum(np.where(np.isfinite(raw), raw, 0.0), 0.0)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    max_observed = float(np.max(observed)) if observed.size else scale
    penalty = max_observed + 20.0 * scale

    working = np.where(valid, distances, penalty)
    np.fill_diagonal(working, penalty)

    # Directed, row-local distance evidence.
    row_min = np.min(working, axis=1)
    row_min = np.maximum(row_min, 0.0)

    row_spread = np.median(
        np.abs(working - row_min[:, None]),
        axis=1,
    )
    row_spread = np.maximum(row_spread, 0.25 * scale)

    distance_signal = np.exp(
        -np.clip(
            (working - row_min[:, None]) / row_spread[:, None],
            0.0,
            25.0,
        )
    )
    distance_signal[~valid] = 0.0
    np.fill_diagonal(distance_signal, 0.0)

    # Directed local-rank evidence.
    ranks = np.empty((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(working[i], kind="stable")
        ranks[i, order] = np.arange(n, dtype=float)

    rank_temperature = max(0.35 * max(n - 1, 1), 1.0)
    rank_signal = np.exp(-ranks / rank_temperature)
    rank_signal[~valid] = 0.0
    np.fill_diagonal(rank_signal, 0.0)

    # Several deterministic nearest-neighbor tours provide sampled-tour evidence.
    start_order = np.argsort(np.min(working, axis=1), kind="stable")
    starts = start_order[: min(n, 8)]

    tour_counts = np.zeros((n, n), dtype=float)
    tours = []
    tour_costs = []

    for start in starts:
        tour = [int(start)]
        unvisited = np.ones(n, dtype=bool)
        unvisited[start] = False
        feasible = True

        while np.any(unvisited):
            current = tour[-1]
            candidates = np.flatnonzero(unvisited)
            next_node = candidates[
                np.argmin(working[current, candidates])
            ]

            if not valid[current, next_node]:
                feasible = False
                break

            tour.append(int(next_node))
            unvisited[next_node] = False

        if not feasible or len(tour) != n or not valid[tour[-1], tour[0]]:
            continue

        tour_array = np.asarray(tour, dtype=int)
        next_array = np.roll(tour_array, -1)
        edges = np.column_stack((tour_array, next_array))

        for a, b in edges:
            tour_counts[a, b] += 1.0

        cost = float(np.sum(working[edges[:, 0], edges[:, 1]]))
        tours.append(tour_array)
        tour_costs.append(cost)

    if tours:
        tour_signal = tour_counts / float(len(tours))
        best_tour = tours[int(np.argmin(tour_costs))]
    else:
        tour_signal = np.zeros((n, n), dtype=float)
        best_tour = None

    # Bounded 2-opt evidence from the best sampled tour.
    opt_signal = np.zeros((n, n), dtype=float)

    if best_tour is not None and n >= 4:
        symmetric = 0.5 * (working + working.T)
        np.fill_diagonal(symmetric, penalty)

        max_gain = max(2.0 * scale, np.finfo(float).eps)
        tour = best_tour.copy()

        for i in range(n - 1):
            a = int(tour[i])
            b = int(tour[(i + 1) % n])

            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue

                c = int(tour[j])
                d = int(tour[(j + 1) % n])

                gain = (
                    symmetric[a, b]
                    + symmetric[c, d]
                    - symmetric[a, c]
                    - symmetric[b, d]
                )

                if gain > 0.0:
                    bounded_gain = np.tanh(gain / max_gain)

                    opt_signal[a, c] = max(opt_signal[a, c], bounded_gain)
                    opt_signal[b, d] = max(opt_signal[b, d], bounded_gain)

                    # Keep this auxiliary structural signal orientation-neutral
                    # without symmetrizing the main directed evidence.
                    opt_signal[c, a] = max(opt_signal[c, a], 0.5 * bounded_gain)
                    opt_signal[d, b] = max(opt_signal[d, b], 0.5 * bounded_gain)

    opt_signal[~valid] = 0.0
    np.fill_diagonal(opt_signal, 0.0)

    # Preserve asymmetric evidence and add only a small exploration floor.
    score = (
        0.52 * distance_signal
        + 0.20 * rank_signal
        + 0.18 * tour_signal
        + 0.10 * opt_signal
    )

    score[~valid] = 0.0
    score = np.clip(score, 0.0, 1.0)

    exploration_floor = 0.035
    result = np.where(
        valid,
        exploration_floor + (1.0 - exploration_floor) * score,
        0.0,
    )
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
