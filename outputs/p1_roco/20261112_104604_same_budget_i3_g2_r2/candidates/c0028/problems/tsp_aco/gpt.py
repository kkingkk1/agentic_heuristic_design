import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite_nonnegative = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)

    valid = finite_nonnegative & finite_nonnegative.T & off_diagonal
    if not np.any(valid):
        return scores

    symmetric_distance = 0.5 * (
        distance_matrix + distance_matrix.T
    )

    participation = np.zeros((n, n), dtype=float)
    early_selection = np.zeros((n, n), dtype=float)
    completed_cycles = 0.0

    for start in range(n):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        route = [start]
        current = start

        while len(route) < n:
            candidates = np.flatnonzero(valid[current] & ~visited)
            if candidates.size == 0:
                break

            feasible_closures = candidates[
                valid[candidates, start] & (candidates != start)
            ]

            if len(route) == n - 1 and feasible_closures.size:
                candidates = feasible_closures

            best_candidate = None
            best_key = None

            for candidate in candidates:
                onward = np.flatnonzero(
                    valid[candidate] & ~visited
                )
                onward = onward[onward != candidate]

                if onward.size:
                    continuation = float(
                        np.min(symmetric_distance[candidate, onward])
                    )
                elif len(route) == n - 1 and valid[candidate, start]:
                    continuation = float(
                        symmetric_distance[candidate, start]
                    )
                else:
                    continuation = np.inf

                key = (
                    float(symmetric_distance[current, candidate]),
                    continuation,
                    int(candidate),
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best_candidate = int(candidate)

            if best_candidate is None:
                break

            previous = current
            current = best_candidate
            visited[current] = True
            route.append(current)

            edge_value = 1.0
            if len(route) <= max(2, n // 2):
                edge_value = 1.5

            participation[previous, current] += edge_value
            participation[current, previous] += edge_value
            early_selection[previous, current] += 1.0
            early_selection[current, previous] += 1.0

        if len(route) == n and valid[current, start]:
            participation[current, start] += 2.0
            participation[start, current] += 2.0
            completed_cycles += 1.0

    if completed_cycles > 0.0:
        participation /= completed_cycles
        early_selection /= float(n)
        scores = participation + 0.25 * early_selection
    else:
        total_paths = float(max(n, 1))
        scores = participation / total_paths

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    scores[~np.isfinite(scores)] = 0.0
    return scores.astype(float, copy=False)
