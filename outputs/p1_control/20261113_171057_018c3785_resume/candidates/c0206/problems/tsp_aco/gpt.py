import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based directed TSP edge priors with two-step lookahead."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    safe = np.where(valid, d, np.inf)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    positions = np.arange(n, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = positions[None, :]
    col_rank[col_order, np.arange(n)[None, :]] = positions[:, None]

    row_rank = np.where(valid, row_rank, np.inf)
    col_rank = np.where(valid, col_rank, np.inf)

    row_width = max(float(n - 1), 1.0)
    ordinal_quality = (
        1.0 / (1.0 + row_rank)
        + 1.0 / (1.0 + col_rank)
    ) / 2.0
    ordinal_quality /= np.max(ordinal_quality[valid])

    neighborhood_size = min(3, n - 1)
    row_neighbors = row_order[:, :neighborhood_size]

    neighborhood_mask = np.zeros((n, n), dtype=bool)
    neighborhood_mask[
        np.arange(n)[:, None],
        row_neighbors,
    ] = True
    neighborhood_mask &= valid

    destination_popularity = neighborhood_mask.sum(axis=0).astype(float)
    popularity_penalty = 1.0 / np.sqrt(
        1.0 + destination_popularity
    )

    continuation = np.min(safe, axis=1)
    continuation_scale = np.where(
        np.isfinite(continuation),
        continuation,
        0.0,
    )

    if n > 2:
        successor_mask = valid.copy()
        successor_mask[np.arange(n), np.argmin(safe, axis=1)] = False
        alternative = np.min(
            np.where(successor_mask, safe, np.inf),
            axis=1,
        )
        continuation_quality = np.divide(
            continuation_scale + np.where(np.isfinite(alternative), alternative, continuation_scale),
            2.0,
            out=np.zeros(n, dtype=float),
        )
    else:
        continuation_quality = continuation_scale

    finite_continuations = continuation_quality[np.isfinite(continuation_quality)]
    baseline = (
        float(np.median(finite_continuations))
        if finite_continuations.size
        else 1.0
    )
    baseline = max(baseline, np.finfo(float).eps)

    continuation_bonus = 1.0 / (
        1.0 + continuation_quality / baseline
    )

    score = ordinal_quality.copy()
    score *= popularity_penalty[None, :]
    score *= continuation_bonus[None, :]
    score *= np.where(neighborhood_mask, 1.35, 0.55)

    if n > 2:
        two_step = np.zeros((n, n), dtype=float)
        for i in range(n):
            candidates = np.flatnonzero(valid[i])
            for j in candidates:
                onward = np.flatnonzero(valid[j])
                onward = onward[onward != i]
                if onward.size:
                    best = np.min(d[j, onward])
                    two_step[i, j] = 1.0 / (
                        1.0 + (d[i, j] + best) / baseline
                    )
        score *= 0.75 + 0.5 * two_step

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
