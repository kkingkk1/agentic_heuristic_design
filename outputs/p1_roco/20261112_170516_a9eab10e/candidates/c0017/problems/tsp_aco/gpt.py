import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros_like(distances, dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    scores = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return scores

    safe_distances = np.where(valid, distances, np.inf)

    inverse_distance = np.zeros_like(distances, dtype=float)
    inverse_distance[valid] = 1.0 / safe_distances[valid]

    row_scale = np.zeros(n, dtype=float)
    for i in range(n):
        row_values = inverse_distance[i, valid[i]]
        if row_values.size:
            row_scale[i] = np.median(row_values)

    accessibility = np.zeros_like(distances, dtype=float)
    for i in range(n):
        if row_scale[i] > 0:
            accessibility[i, valid[i]] = (
                inverse_distance[i, valid[i]] / row_scale[i]
            )

    regret = np.zeros_like(distances, dtype=float)
    for i in range(n):
        candidates = np.sort(safe_distances[i])
        candidates = candidates[np.isfinite(candidates)]
        if candidates.size >= 2:
            best = candidates[0]
            second_best = candidates[1]
            for j in range(n):
                if valid[i, j]:
                    regret[i, j] = max(
                        0.0,
                        (distances[i, j] - best) / max(second_best, best, 1e-12),
                    )

    pair_accessibility = accessibility + accessibility.T
    pair_regret = regret + regret.T
    base = inverse_distance + inverse_distance.T

    finite_base = base[base > 0]
    scale = np.median(finite_base) if finite_base.size else 1.0
    base = base / max(scale, 1e-12)

    scores = base * (1.0 + 0.35 * pair_accessibility)
    scores += 0.25 * pair_regret
    scores *= valid & valid.T

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores.astype(float)
