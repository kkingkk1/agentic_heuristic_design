import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if np.any(np.isnan(distances)) or np.any(distances < 0):
        raise ValueError(
            "distance_matrix must contain non-negative, non-NaN distances"
        )

    finite = np.isfinite(distances)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    finite_values = distances[finite & (distances > 0)]
    global_scale = (
        float(np.median(finite_values)) if finite_values.size else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    work = np.where(valid, 0.5 * (distances + distances.T), np.inf)
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps

    # Endpoint-local scales make shortness comparable across heterogeneous regions.
    ordered = np.sort(work, axis=1)
    k = min(max(1, int(np.sqrt(n))), n - 1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    shortness = np.exp(
        -np.minimum(work / (pair_scale + eps), 50.0)
    )
    shortness = np.where(valid, shortness, 0.0)

    # Mutual rank preference favors edges preferred by both endpoints.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks = np.maximum(ranks, 1.0)

    rank_scale = max(1.0, np.sqrt(n))
    mutual_rank = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )
    mutual_rank = np.where(valid, mutual_rank, 0.0)

    # Estimate whether a direct edge is competitive with its best two-hop route.
    detour_resistance = np.full((n, n), 0.5, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if not valid[i, j]:
                continue

            intermediates = np.ones(n, dtype=bool)
            intermediates[[i, j]] = False
            candidates = work[i, :] + work[:, j]
            candidates = candidates[intermediates]

            if candidates.size and np.any(np.isfinite(candidates)):
                best_detour = float(np.min(candidates))
                direct = float(work[i, j])

                if np.isfinite(best_detour):
                    resistance = best_detour / (
                        direct + best_detour + eps
                    )
                else:
                    resistance = 1.0
            else:
                resistance = 1.0

            detour_resistance[i, j] = resistance
            detour_resistance[j, i] = resistance

    detour_resistance = np.where(valid, detour_resistance, 0.0)

    score = (
        np.power(shortness + eps, 0.50)
        * np.power(mutual_rank + eps, 0.30)
        * np.power(detour_resistance + eps, 0.20)
    )
    score = 0.5 * (score + score.T)
    score = np.where(valid, score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
