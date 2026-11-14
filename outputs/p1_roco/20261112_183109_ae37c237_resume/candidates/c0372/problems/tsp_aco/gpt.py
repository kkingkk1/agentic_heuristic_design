import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced local-distance and mild long-term TSP edge priorities."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    k = min(4, n - 1)

    row_sorted = np.sort(work, axis=1)
    row_nearest = row_sorted[:, :k]
    row_finite = np.isfinite(row_nearest)
    row_counts = row_finite.sum(axis=1)
    row_sum = np.where(row_finite, row_nearest, 0.0).sum(axis=1)
    row_scale = np.divide(
        row_sum,
        np.maximum(row_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=row_counts > 0,
    )
    row_scale = np.maximum(row_scale, eps)

    col_work = work.T
    col_sorted = np.sort(col_work, axis=1)
    col_nearest = col_sorted[:, :k]
    col_finite = np.isfinite(col_nearest)
    col_counts = col_finite.sum(axis=1)
    col_sum = np.where(col_finite, col_nearest, 0.0).sum(axis=1)
    col_scale = np.divide(
        col_sum,
        np.maximum(col_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=col_counts > 0,
    )
    col_scale = np.maximum(col_scale, eps)

    row_best = row_sorted[:, 0]
    row_second = (
        row_sorted[:, 1] if n > 2 else row_best.copy()
    )
    row_best = np.where(np.isfinite(row_best), row_best, row_scale)
    row_second = np.where(
        np.isfinite(row_second),
        row_second,
        np.maximum(row_best, row_scale),
    )

    col_best = col_sorted[:, 0]
    col_second = (
        col_sorted[:, 1] if n > 2 else col_best.copy()
    )
    col_best = np.where(np.isfinite(col_best), col_best, col_scale)
    col_second = np.where(
        np.isfinite(col_second),
        col_second,
        np.maximum(col_best, col_scale),
    )

    row_regret = np.clip(
        (row_second - row_best) / np.maximum(row_second, eps),
        0.0,
        1.0,
    )
    col_regret = np.clip(
        (col_second - col_best) / np.maximum(col_second, eps),
        0.0,
        1.0,
    )

    row_ratio = np.divide(
        d,
        row_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    geometric_scale = np.sqrt(
        row_scale[:, None] * col_scale[None, :]
    )
    geometric_ratio = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        distance_quality = np.exp(
            -np.clip(geometric_ratio, 0.0, 50.0) / 0.70
        )
        row_quality = np.exp(
            -np.clip(row_ratio, 0.0, 50.0) / 0.80
        )
        col_quality = np.exp(
            -np.clip(col_ratio, 0.0, 50.0) / 0.80
        )
        mutual_quality = np.sqrt(
            np.maximum(row_quality, 0.0)
            * np.maximum(col_quality, 0.0)
        )
        global_quality = np.exp(
            -np.clip(d / global_scale, 0.0, 50.0) / 1.60
        )

    score = (
        0.72 * distance_quality
        + 0.18 * mutual_quality
        + 0.06 * global_quality
        + 0.04 * np.sqrt(
            np.maximum(row_quality, 0.0)
            * np.maximum(col_quality, 0.0)
        )
    )

    urgency = (
        1.0
        + 0.06 * row_regret[:, None]
        + 0.04 * col_regret[None, :]
    )
    score *= urgency

    row_near = np.sum(
        valid & (d <= row_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_near = np.sum(
        valid & (d <= col_second[None, :]),
        axis=0,
        dtype=float,
    )

    scarcity = (
        1.0 / np.sqrt(np.maximum(row_near, 1.0))[:, None]
        + 1.0 / np.sqrt(np.maximum(col_near, 1.0))[None, :]
    )
    scarcity /= max(float(np.max(scarcity)), 1.0)
    score *= 1.0 + 0.08 * scarcity

    score = np.where(valid, np.maximum(score, 0.0), 0.0)

    symmetric_input = (
        np.array_equal(valid, valid.T)
        and np.allclose(
            np.where(valid, d, 0.0),
            np.where(valid.T, d.T, 0.0),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    if symmetric_input:
        score = np.sqrt(
            np.maximum(score, 0.0)
            * np.maximum(score.T, 0.0)
        )

    degree = valid.sum(axis=1, keepdims=True).astype(float)
    exploration_prior = np.divide(
        valid.astype(float),
        np.maximum(degree, 1.0),
    )
    score = 0.96 * score + 0.04 * exploration_prior
    score = np.where(valid, np.maximum(score, 0.0), 0.0)
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
