import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, bidirectional-regret priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    result = np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    masked = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    finite = d[valid]

    if positive.size:
        distance_scale = float(np.median(positive))
    elif finite.size:
        distance_scale = float(np.median(finite))
    else:
        distance_scale = 1.0

    distance_scale = max(distance_scale, eps)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_candidates = np.partition(masked, kth=1, axis=1)[:, :2]
    col_candidates = np.partition(masked, kth=1, axis=0)[:2, :]

    row_min = row_candidates[:, 0]
    row_second = row_candidates[:, 1]
    col_min = col_candidates[0, :]
    col_second = col_candidates[1, :]

    row_min_safe = np.where(row_count > 0, row_min, 0.0)
    col_min_safe = np.where(col_count > 0, col_min, 0.0)

    row_delta = np.zeros_like(d)
    col_delta = np.zeros_like(d)

    np.subtract(
        d,
        row_min_safe[:, None],
        out=row_delta,
        where=valid,
    )
    np.subtract(
        d,
        col_min_safe[None, :],
        out=col_delta,
        where=valid,
    )

    np.maximum(row_delta, 0.0, out=row_delta, where=valid)
    np.maximum(col_delta, 0.0, out=col_delta, where=valid)

    row_gap = np.where(
        row_count >= 2,
        np.maximum(row_second - row_min_safe, 0.0),
        distance_scale,
    )
    col_gap = np.where(
        col_count >= 2,
        np.maximum(col_second - col_min_safe, 0.0),
        distance_scale,
    )

    positive_row_gaps = row_gap[
        np.isfinite(row_gap) & (row_gap > 0.0)
    ]
    positive_col_gaps = col_gap[
        np.isfinite(col_gap) & (col_gap > 0.0)
    ]

    row_gap_scale = (
        float(np.median(positive_row_gaps))
        if positive_row_gaps.size
        else distance_scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gaps))
        if positive_col_gaps.size
        else distance_scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        regret_exponent = -0.5 * (
            row_delta / row_gap_scale
            + col_delta / col_gap_scale
        )
        regret = np.exp(np.clip(regret_exponent, -60.0, 0.0))

        temperature = max(0.22 * distance_scale, eps)
        logits = 0.5 * (row_delta + col_delta) / temperature
        affinity = np.exp(-np.clip(logits, 0.0, 60.0))

    regret[~valid] = 0.0
    affinity[~valid] = 0.0

    if n <= 64:
        max_steps = 48
    elif n <= 256:
        max_steps = 24
    else:
        max_steps = 12

    balanced = affinity.copy()

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(max_steps):
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

            balanced[~np.isfinite(balanced)] = 0.0

            if np.max(np.abs(balanced - previous)) < 1e-10:
                break

    score = balanced * np.power(regret + eps, 0.9)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
