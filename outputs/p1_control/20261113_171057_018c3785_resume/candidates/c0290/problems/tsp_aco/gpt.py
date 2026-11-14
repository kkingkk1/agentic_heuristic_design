import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware priors for directed TSP edges."""
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

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    distance_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    distance_scale = max(distance_scale, eps)

    inf = np.inf
    masked = np.where(valid, d, inf)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(masked, axis=1)
    col_min = np.min(masked, axis=0)

    row_min = np.where(row_count > 0, row_min, 0.0)
    col_min = np.where(col_count > 0, col_min, 0.0)

    row_delta = np.maximum(d - row_min[:, None], 0.0)
    col_delta = np.maximum(d - col_min[None, :], 0.0)
    row_delta[~valid] = 0.0
    col_delta[~valid] = 0.0

    row_second = np.full(n, inf, dtype=float)
    col_second = np.full(n, inf, dtype=float)

    row_mask = row_count >= 2
    if np.any(row_mask):
        row_values = masked[row_mask]
        row_second[row_mask] = np.partition(
            row_values,
            kth=1,
            axis=1,
        )[:, 1]

    col_mask = col_count >= 2
    if np.any(col_mask):
        col_values = masked[:, col_mask]
        col_second[col_mask] = np.partition(
            col_values,
            kth=1,
            axis=0,
        )[1, :]

    row_gap = np.where(
        row_mask,
        np.maximum(row_second - row_min, 0.0),
        distance_scale,
    )
    col_gap = np.where(
        col_mask,
        np.maximum(col_second - col_min, 0.0),
        distance_scale,
    )

    row_positive_gap = row_gap[
        np.isfinite(row_gap) & (row_gap > 0.0)
    ]
    col_positive_gap = col_gap[
        np.isfinite(col_gap) & (col_gap > 0.0)
    ]

    row_gap_scale = (
        float(np.median(row_positive_gap))
        if row_positive_gap.size
        else distance_scale
    )
    col_gap_scale = (
        float(np.median(col_positive_gap))
        if col_positive_gap.size
        else distance_scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    regret_exponent = 0.5 * (
        row_delta / row_gap_scale
        + col_delta / col_gap_scale
    )
    regret = np.exp(-np.clip(regret_exponent, 0.0, 60.0))
    regret[~valid] = 0.0

    temperature = max(0.20 * distance_scale, eps)
    gibbs_logits = np.clip(
        0.5 * (row_delta + col_delta) / temperature,
        0.0,
        60.0,
    )
    affinity = np.exp(-gibbs_logits)
    affinity[~valid] = 0.0

    balanced = affinity
    for _ in range(48):
        previous = balanced.copy()

        row_sum = balanced.sum(axis=1, keepdims=True)
        np.divide(
            balanced,
            row_sum,
            out=balanced,
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        np.divide(
            balanced,
            col_sum,
            out=balanced,
            where=col_sum > 0.0,
        )

        if np.max(np.abs(balanced - previous)) < 1e-10:
            break

    score = balanced * np.power(regret + eps, 1.1)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
