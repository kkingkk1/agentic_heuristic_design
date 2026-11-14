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
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = raw[valid & (raw > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    observed = raw[valid]
    high = float(np.quantile(observed, 0.95)) if observed.size else scale
    penalty = max(high, scale) + 20.0 * scale

    distances = np.where(valid, raw, penalty)
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps

    # Reciprocal-distance evidence with robust saturation.
    reciprocal = 1.0 / (1.0 + distances / scale)
    reciprocal[~valid] = 0.0
    reciprocal[diagonal] = 0.0

    # Directed row-wise competition signal.
    rank_signal = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[
            np.argsort(distances[i, candidates], kind="stable")
        ]
        if order.size == 1:
            rank_signal[i, order[0]] = 1.0
            continue

        ranks = np.empty(order.size, dtype=float)
        ranks[np.arange(order.size)] = np.arange(order.size)
        rank_signal[i, order] = 1.0 - ranks / (order.size - 1.0)

    # Robustly normalize base components without forcing symmetry.
    def normalize(component: np.ndarray) -> np.ndarray:
        values = component[valid]
        if values.size == 0:
            return np.zeros_like(component)

        low = float(np.quantile(values, 0.05))
        high_value = float(np.quantile(values, 0.95))
        width = max(high_value - low, eps)

        normalized = (component - low) / width
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized[~valid] = 0.0
        return normalized

    reciprocal = normalize(reciprocal)
    rank_signal = normalize(rank_signal)

    # Build several deterministic nearest-neighbor tours.
    tour_frequency = np.zeros((n, n), dtype=float)
    tours = []

    for start in range(n):
        for offset in (0, 1):
            tour = [start]
            unvisited = np.ones(n, dtype=bool)
            unvisited[start] = False
            current = start
            feasible = True

            while np.any(unvisited):
                candidates = np.flatnonzero(unvisited & valid[current])
                if candidates.size == 0:
                    feasible = False
                    break

                candidate_distances = distances[current, candidates]
                order = np.argsort(candidate_distances, kind="stable")
                chosen = candidates[order[min(offset, order.size - 1)]]

                tour.append(int(chosen))
                unvisited[chosen] = False
                current = int(chosen)

            if feasible and len(tour) == n and valid[tour[-1], tour[0]]:
                tours.append(tour)
                for a, b in zip(tour, tour[1:] + tour[:1]):
                    tour_frequency[a, b] += 1.0

    if tours:
        tour_frequency /= float(len(tours))
    tour_signal = normalize(tour_frequency)

    # Use 2-opt evidence only when the instance is sufficiently symmetric.
    finite_pairs = valid & valid.T
    if np.any(finite_pairs):
        asymmetry = np.abs(raw - raw.T)
        pair_scale = np.maximum(np.abs(raw), np.abs(raw.T)) + scale
        asymmetry_ratio = float(
            np.median((asymmetry[finite_pairs] / pair_scale[finite_pairs]))
        )
    else:
        asymmetry_ratio = 1.0

    two_opt_support = np.zeros((n, n), dtype=float)

    if tours and asymmetry_ratio <= 0.15:
        for tour in tours:
            for left in range(n - 2):
                a = tour[left]
                b = tour[(left + 1) % n]

                for right in range(left + 2, n):
                    if left == 0 and right == n - 1:
                        continue

                    c = tour[right]
                    d = tour[(right + 1) % n]

                    if not (
                        valid[a, c]
                        and valid[b, d]
                        and valid[c, a]
                        and valid[d, b]
                    ):
                        continue

                    old_cost = distances[a, b] + distances[c, d]
                    new_cost = distances[a, c] + distances[b, d]
                    gain = old_cost - new_cost

                    if gain > 0.0:
                        weight = gain / (scale + gain)
                        two_opt_support[a, c] += weight
                        two_opt_support[b, d] += weight

        if tours:
            two_opt_support /= float(len(tours))

    two_opt_signal = normalize(two_opt_support)

    # Small floor applies only to valid edges, preserving global exploration.
    floor = 0.02
    score = (
        0.42 * reciprocal
        + 0.28 * rank_signal
        + 0.18 * tour_signal
        + 0.12 * two_opt_signal
    )
    score = (1.0 - floor) * score + floor
    score[~valid] = 0.0
    score[diagonal] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
