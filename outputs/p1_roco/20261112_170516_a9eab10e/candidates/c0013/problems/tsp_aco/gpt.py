import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n < 2:
        return np.zeros_like(distances, dtype=float)

    d = distances.copy()
    finite_positive = np.isfinite(d) & (d > 0)
    fallback = np.median(d[finite_positive]) if np.any(finite_positive) else 1.0
    d[~np.isfinite(d)] = fallback * 10.0
    d[d < 0] = fallback * 10.0
    np.fill_diagonal(d, 0.0)

    scale = np.median(d[finite_positive]) if np.any(finite_positive) else fallback
    scale = max(float(scale), np.finfo(float).eps)

    local = np.zeros((n, n), dtype=float)
    valid = d > 0
    local[valid] = 1.0 / (d[valid] + 1e-12)
    local /= max(float(local.max()), np.finfo(float).eps)
    np.fill_diagonal(local, 0.0)

    seed_value = int(np.abs(np.sum(d / scale) * 1_000_003)) % (2**32)
    rng = np.random.default_rng(seed_value)

    samples = max(32, min(8 * n, 256))
    edge_scores = np.zeros((n, n), dtype=float)
    tour_lengths = []

    for sample in range(samples):
        start = sample % n
        unvisited = np.ones(n, dtype=bool)
        unvisited[start] = False
        tour = [start]
        current = start

        while np.any(unvisited):
            candidates = np.flatnonzero(unvisited)
            distances_to_candidates = d[current, candidates]
            attractiveness = 1.0 / (distances_to_candidates + 1e-12)
            attractiveness *= 1.0 + 0.25 * local[current, candidates]

            temperature = 0.55 + 0.45 * ((sample % 7) / 6.0)
            probabilities = attractiveness ** (1.0 / temperature)
            probabilities /= probabilities.sum()

            next_node = int(rng.choice(candidates, p=probabilities))
            tour.append(next_node)
            unvisited[next_node] = False
            current = next_node

        tour_array = np.asarray(tour, dtype=int)
        length = float(
            np.sum(d[tour_array, np.roll(tour_array, -1)])
        )
        if np.isfinite(length) and length > 0:
            tour_lengths.append((tour_array, length))

    if tour_lengths:
        lengths = np.asarray([length for _, length in tour_lengths])
        reference = max(float(np.median(lengths)), np.finfo(float).eps)

        for tour, length in tour_lengths:
            weight = np.exp(-max(length / reference - 1.0, -20.0))
            for a, b in zip(tour, np.roll(tour, -1)):
                edge_scores[a, b] += weight
                edge_scores[b, a] += weight

        if edge_scores.max() > 0:
            edge_scores /= edge_scores.max()

    result = 0.4 * local + 0.6 * edge_scores
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    return result.astype(float, copy=False)
