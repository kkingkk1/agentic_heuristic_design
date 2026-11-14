import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.where(finite, raw, 0.0)
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    edge_values = distances[valid]
    positive = edge_values[edge_values > 0.0]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    finite_edges = edge_values[np.isfinite(edge_values)]
    if finite_edges.size:
        upper = float(np.percentile(finite_edges, 95.0))
    else:
        upper = global_scale

    cap = max(upper + 2.0 * global_scale, global_scale)
    distances = np.minimum(distances, cap)
    distances[~valid] = cap
    np.fill_diagonal(distances, cap)

    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, -1, dtype=int)
    ranks = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[
            np.argsort(distances[i, candidates], kind="stable")
        ]
        nearest[i] = int(order[0])

        best = float(distances[i, order[0]])
        deviations = distances[i, order] - best

        if deviations.size > 1:
            local_scale[i] = max(
                float(np.median(deviations[1:])),
                0.25 * global_scale,
                np.finfo(float).eps,
            )
        else:
            local_scale[i] = max(
                0.25 * global_scale,
                np.finfo(float).eps,
            )

        if order.size > 1:
            ranks[i, order] = 1.0 - (
                np.arange(order.size, dtype=float) / float(order.size - 1)
            )
        else:
            ranks[i, order] = 1.0

    temperature = np.sqrt(local_scale[:, None] * local_scale[None, :])
    temperature = np.maximum(temperature, 0.40 * global_scale)

    affinity = np.exp(-distances / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if np.isfinite(affinity_max) and affinity_max > 0.0:
        affinity /= affinity_max

    rank_score = np.sqrt(ranks * ranks.T)

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0

    mutual = 0.5 * (mutual + mutual.T)

    score = (
        0.84 * affinity
        + 0.12 * rank_score
        + 0.04 * mutual
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    row_sum = score.sum(axis=1)
    positive_rows = row_sum > np.finfo(float).eps

    if np.any(positive_rows):
        row_mean = np.zeros(n, dtype=float)
        row_mean[positive_rows] = (
            row_sum[positive_rows] / np.maximum(valid.sum(axis=1)[positive_rows], 1)
        )

        scale = np.ones(n, dtype=float)
        reference = max(float(np.median(row_mean[positive_rows])), np.finfo(float).eps)
        scale[positive_rows] = np.sqrt(
            np.clip(row_mean[positive_rows] / reference, 0.5, 2.0)
        )

        score = score / np.sqrt(scale[:, None] * scale[None, :])

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
