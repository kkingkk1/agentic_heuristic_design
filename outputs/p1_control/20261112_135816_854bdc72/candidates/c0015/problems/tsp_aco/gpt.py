import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a stochastic-sampling-based prior for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(d, 0.0).copy()
    np.fill_diagonal(d, 0.0)

    positive = d[d > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = max(scale * 1e-9, 1e-12)

    attractiveness = 1.0 / (d + eps)
    np.fill_diagonal(attractiveness, 0.0)
    if np.max(attractiveness) > 0:
        direct_prior = attractiveness / np.max(attractiveness)
    else:
        direct_prior = np.zeros((n, n), dtype=float)

    row_scale = np.maximum(np.mean(d + np.eye(n) * scale, axis=1), eps)
    normalized_distance = d / np.sqrt(np.outer(row_scale, row_scale))

    rng = np.random.default_rng(20261112 + 31 * n)
    support = np.zeros((n, n), dtype=float)
    tour_lengths = []

    temperatures = (0.20, 0.40, 0.80, 1.60)
    samples_per_temperature = max(8, min(24, 96 // max(n, 1)))

    for temperature in temperatures:
        for _ in range(samples_per_temperature):
            start = int(rng.integers(n))
            tour = [start]
            unvisited = np.ones(n, dtype=bool)
            unvisited[start] = False

            while np.any(unvisited):
                current = tour[-1]
                candidates = np.flatnonzero(unvisited)
                candidate_distances = normalized_distance[current, candidates]

                if candidates.size > 1:
                    lookahead = np.min(
                        normalized_distance[np.ix_(candidates, candidates)],
                        axis=1,
                    )
                else:
                    lookahead = np.zeros(1, dtype=float)

                score = (
                    -candidate_distances / max(temperature, 1e-6)
                    -0.35 * lookahead / max(temperature, 1e-6)
                )
                score -= np.max(score)
                probabilities = np.exp(score)
                probabilities /= np.sum(probabilities)

                next_node = int(rng.choice(candidates, p=probabilities))
                tour.append(next_node)
                unvisited[next_node] = False

            tour_array = np.asarray(tour, dtype=int)
            closed_tour = np.concatenate((tour_array, tour_array[:1]))
            length = float(np.sum(d[closed_tour[:-1], closed_tour[1:]]))
            tour_lengths.append(length)

            for u, v in zip(closed_tour[:-1], closed_tour[1:]):
                support[u, v] += 1.0

    tour_lengths = np.asarray(tour_lengths, dtype=float)
    if tour_lengths.size:
        reference = max(float(np.median(tour_lengths)), eps)
        weights = np.exp(
            -(tour_lengths - np.min(tour_lengths)) / max(reference, eps)
        )
        weighted_support = np.zeros_like(support)
        index = 0

        for temperature in temperatures:
            for _ in range(samples_per_temperature):
                start = int(rng.integers(n))
                tour = [start]
                unvisited = np.ones(n, dtype=bool)
                unvisited[start] = False

                while np.any(unvisited):
                    current = tour[-1]
                    candidates = np.flatnonzero(unvisited)
                    candidate_distances = normalized_distance[current, candidates]
                    score = -candidate_distances / max(temperature, 1e-6)
                    score -= np.max(score)
                    probabilities = np.exp(score)
                    probabilities /= np.sum(probabilities)
                    next_node = int(rng.choice(candidates, p=probabilities))
                    tour.append(next_node)
                    unvisited[next_node] = False

                closed = np.asarray(tour + [tour[0]], dtype=int)
                weight = weights[index]
                for u, v in zip(closed[:-1], closed[1:]):
                    weighted_support[u, v] += weight
                index += 1

        if np.max(weighted_support) > 0:
            weighted_support /= np.max(weighted_support)
    else:
        weighted_support = np.zeros_like(support)

    prior = 0.55 * direct_prior + 0.45 * weighted_support
    prior = 0.5 * (prior + prior.T)
    np.fill_diagonal(prior, 0.0)

    maximum = np.max(prior)
    if maximum > 0:
        prior /= maximum

    return prior.astype(float, copy=False)
