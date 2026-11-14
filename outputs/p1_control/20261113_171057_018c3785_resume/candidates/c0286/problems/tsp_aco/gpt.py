import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return finite, competition-aware edge priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    tiny = np.finfo(float).eps
    scale = max(scale, tiny)

    safe_d = np.where(valid, d, np.inf)
    normalized = np.divide(
        safe_d,
        scale,
        out=np.full_like(safe_d, np.inf, dtype=float),
        where=valid,
    )

    # Smooth inverse-distance affinity, stable for zero-length edges.
    base = np.zeros_like(d, dtype=float)
    base[valid] = 1.0 / (1.0 + normalized[valid])

    row_sum = np.sum(base, axis=1, keepdims=True)
    col_sum = np.sum(base, axis=0, keepdims=True)

    row_prob = np.divide(
        base,
        row_sum,
        out=np.zeros_like(base),
        where=row_sum > 0.0,
    )
    col_prob = np.divide(
        base,
        col_sum,
        out=np.zeros_like(base),
        where=col_sum > 0.0,
    )

    # Geometric fusion rewards edges that are good from both directions.
    affinity = np.sqrt(row_prob * col_prob)

    # Estimate edge scarcity from row and column competition.
    row_competition = np.max(row_prob, axis=1, keepdims=True)
    col_competition = np.max(col_prob, axis=0, keepdims=True)

    scarcity = np.sqrt(
        np.divide(
            row_prob,
            row_competition,
            out=np.zeros_like(row_prob),
            where=row_competition > 0.0,
        )
        * np.divide(
            col_prob,
            col_competition,
            out=np.zeros_like(col_prob),
            where=col_competition > 0.0,
        )
    )

    # Regret emphasizes edges whose removal leaves a substantially worse option.
    row_sorted = np.sort(np.where(valid, normalized, np.inf), axis=1)
    col_sorted = np.sort(np.where(valid, normalized, np.inf), axis=0)

    row_best = row_sorted[:, 0]
    col_best = col_sorted[0, :]
    row_second = row_sorted[:, 1]
    col_second = col_sorted[1, :]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    row_gap[~np.isfinite(row_gap)] = 0.0
    col_gap[~np.isfinite(col_gap)] = 0.0

    gap_values = np.concatenate(
        [
            row_gap[row_gap > 0.0],
            col_gap[col_gap > 0.0],
        ]
    )
    gap_scale = float(np.median(gap_values)) if gap_values.size else 1.0
    if not np.isfinite(gap_scale) or gap_scale <= 0.0:
        gap_scale = 1.0

    row_delta = np.maximum(
        np.where(valid, normalized - row_best[:, None], 0.0),
        0.0,
    )
    col_delta = np.maximum(
        np.where(valid, normalized - col_best[None, :], 0.0),
        0.0,
    )

    regret = np.exp(
        -np.clip(
            0.5 * (row_delta + col_delta) / gap_scale,
            0.0,
            60.0,
        )
    )
    regret[~valid] = 0.0

    # Prefer mutually strong edges while retaining local alternatives.
    score = (
        (0.60 * affinity + 0.40 * scarcity)
        * np.power(regret + tiny, 0.8)
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
