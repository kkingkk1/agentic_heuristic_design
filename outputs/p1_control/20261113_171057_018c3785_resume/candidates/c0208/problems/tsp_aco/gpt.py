import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed TSP edge priors from local scarcity and detour advantage."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0.0):
        raise ValueError("distance_matrix must be nonnegative")

    n = distances.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    finite = work[valid]
    positive = finite[finite > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    elif finite.size:
        scale = max(float(np.median(finite)), 1.0)
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    outgoing_best = np.min(work, axis=1)
    incoming_best = np.min(work, axis=0)

    row_margin = np.maximum(work - outgoing_best[:, None], 0.0)
    col_margin = np.maximum(work - incoming_best[None, :], 0.0)

    # Compress large distances while preserving distinctions among short edges.
    robust_cost = np.log1p(np.maximum(work, 0.0) / scale)
    base = 1.0 / (1.0 + robust_cost)
    base[~valid] = 0.0

    # Count how many alternatives are competitive at each endpoint.
    row_threshold = outgoing_best + 0.75 * scale
    col_threshold = incoming_best + 0.75 * scale

    row_competitors = np.sum(
        valid & (work <= row_threshold[:, None]),
        axis=1,
    ).astype(float)
    col_competitors = np.sum(
        valid & (work <= col_threshold[None, :]),
        axis=0,
    ).astype(float)

    row_scarcity = 1.0 / np.sqrt(np.maximum(row_competitors, 1.0))
    col_scarcity = 1.0 / np.sqrt(np.maximum(col_competitors, 1.0))
    scarcity = 1.0 + 0.85 * np.sqrt(
        row_scarcity[:, None] * col_scarcity[None, :]
    )

    # Prefer edges that are simultaneously competitive from both endpoints.
    endpoint_margin = np.sqrt(
        np.exp(-np.minimum(row_margin / scale, 40.0))
        * np.exp(-np.minimum(col_margin / scale, 40.0))
    )

    # Estimate whether a direct edge avoids an expensive two-hop detour.
    best_two_hop = np.full((n, n), np.inf, dtype=float)

    for middle in range(n):
        left = work[:, middle][:, None]
        right = work[middle, :][None, :]
        candidate = left + right

        allowed = valid.copy()
        allowed[:, middle] = False
        allowed[middle, :] = False

        best_two_hop = np.minimum(
            best_two_hop,
            np.where(allowed, candidate, np.inf),
        )

    detour_gain = np.maximum(best_two_hop - work, 0.0)
    detour_gain = np.minimum(detour_gain / scale, 12.0)
    detour_bonus = 1.0 + 0.55 * np.tanh(detour_gain)

    # Penalize edges that are poor from either endpoint without hard pruning.
    endpoint_penalty = np.exp(
        -0.28
        * np.minimum(
            np.sqrt(row_margin * row_margin + col_margin * col_margin)
            / scale,
            40.0,
        )
    )

    score = (
        base
        * endpoint_margin
        * scarcity
        * detour_bonus
        * endpoint_penalty
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    positive_scores = score[valid & (score > 0.0)]
    if positive_scores.size:
        mean_score = float(np.mean(positive_scores))
        if mean_score > 0.0 and np.isfinite(mean_score):
            score /= mean_score

    np.fill_diagonal(score, 0.0)
    return score
