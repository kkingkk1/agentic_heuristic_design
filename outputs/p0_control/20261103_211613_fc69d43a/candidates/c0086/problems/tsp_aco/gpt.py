import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return locally adaptive edge scores for a fixed ant-colony TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    usable = distances[valid]
    if usable.size == 0:
        return scores

    fallback = float(np.median(usable))
    safe = np.where(valid, distances, fallback)
    np.fill_diagonal(safe, np.inf)

    neighbor_count = max(1, min(8, n - 1))
    local_scales = np.empty(n, dtype=float)

    for i in range(n):
        row = safe[i]
        finite_row = row[np.isfinite(row)]
        if finite_row.size == 0:
            local_scales[i] = fallback
        else:
            nearest = np.partition(
                finite_row,
                min(neighbor_count - 1, finite_row.size - 1),
            )[:neighbor_count]
            local_scales[i] = float(np.median(nearest))

    local_scales = np.where(
        np.isfinite(local_scales) & (local_scales > 0),
        local_scales,
        fallback,
    )

    directed_affinity = local_scales[:, None] / safe
    reverse_affinity = local_scales[None, :] / safe

    multiplicative_affinity = np.sqrt(
        np.maximum(directed_affinity, 0.0)
        * np.maximum(reverse_affinity, 0.0)
    )

    neighbor_sets = []
    for i in range(n):
        order = np.argsort(safe[i], kind="stable")
        neighbor_sets.append(set(order[:neighbor_count]))

    compatibility = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if not valid[i, j]:
                continue
            shared = len(neighbor_sets[i].intersection(neighbor_sets[j]))
            union = len(neighbor_sets[i].union(neighbor_sets[j]))
            compatibility[i, j] = shared / union if union else 0.0

    raw = (1.0 / safe) * (
        1.0
        + multiplicative_affinity
        * (0.75 + 0.5 * compatibility)
    )

    raw = np.where(valid, raw, 0.0)
    np.fill_diagonal(raw, 0.0)

    maximum = float(np.max(raw))
    if not np.isfinite(maximum) or maximum <= 0:
        return scores

    directed_scores = raw / maximum
    scores = 0.85 * directed_scores + 0.15 * directed_scores.T
    scores = np.where(valid | valid.T, scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
