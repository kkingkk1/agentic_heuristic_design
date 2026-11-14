import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return two-step lookahead priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    safe = np.where(valid, d, np.inf)

    finite = d[valid]
    scale = float(np.median(finite[finite > 0.0])) if np.any(finite > 0.0) else 1.0
    scale = max(scale, eps)

    # Robust local distance normalization.
    row_mean = np.divide(
        np.where(valid, d, 0.0).sum(axis=1),
        valid.sum(axis=1),
        out=np.full(n, scale, dtype=float),
        where=valid.sum(axis=1) > 0,
    )
    col_mean = np.divide(
        np.where(valid, d, 0.0).sum(axis=0),
        valid.sum(axis=0),
        out=np.full(n, scale, dtype=float),
        where=valid.sum(axis=0) > 0,
    )

    local_scale = np.sqrt(
        np.maximum(row_mean[:, None], eps)
        * np.maximum(col_mean[None, :], eps)
    )

    normalized = np.divide(
        d,
        local_scale,
        out=np.zeros_like(d),
        where=valid,
    )

    # Base preference uses a rational decay, unlike exponential Gibbs scores.
    base = np.divide(
        1.0,
        1.0 + np.maximum(normalized, 0.0),
        out=np.zeros_like(d),
        where=valid,
    )

    # Best continuation after i -> j, and best predecessor before i -> j.
    continuation = np.min(safe, axis=1)
    predecessor = np.min(safe, axis=0)

    next_cost = continuation[None, :]
    prev_cost = predecessor[:, None]

    continuation_quality = np.divide(
        1.0,
        1.0 + next_cost / scale,
        out=np.zeros((1, n), dtype=float),
        where=np.isfinite(next_cost),
    )
    predecessor_quality = np.divide(
        1.0,
        1.0 + prev_cost / scale,
        out=np.zeros((n, 1), dtype=float),
        where=np.isfinite(prev_cost),
    )

    lookahead = np.sqrt(continuation_quality * predecessor_quality)

    # Penalize destinations and sources that attract too much cheap traffic.
    cheapness = np.divide(
        1.0,
        1.0 + np.maximum(d, 0.0) / scale,
        out=np.zeros_like(d),
        where=valid,
    )

    incoming_pressure = cheapness.sum(axis=0)
    outgoing_pressure = cheapness.sum(axis=1)

    incoming_factor = np.divide(
        1.0,
        np.sqrt(1.0 + incoming_pressure[None, :]),
    )
    outgoing_factor = np.divide(
        1.0,
        np.sqrt(1.0 + outgoing_pressure[:, None]),
    )

    # Reward edges participating in a cheap two-step cyclic closure.
    two_step = np.zeros_like(d)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(n):
            left = np.divide(
                1.0,
                1.0 + np.maximum(d[:, k], 0.0) / scale,
                out=np.zeros(n, dtype=float),
                where=valid[:, k],
            )
            right = np.divide(
                1.0,
                1.0 + np.maximum(d[k, :], 0.0) / scale,
                out=np.zeros(n, dtype=float),
                where=valid[k, :],
            )
            two_step += np.outer(left, right)

    closure = np.divide(
        two_step,
        np.maximum(two_step.max(), eps),
        out=np.zeros_like(two_step),
    )

    score = (
        base
        * (0.55 + 0.90 * lookahead)
        * (0.65 + 0.35 * closure)
        * incoming_factor
        * outgoing_factor
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    positive = score[score > 0.0]
    if positive.size:
        floor = max(float(np.percentile(positive, 5)) * 0.08, eps)
        score[valid] = np.maximum(score[valid], floor)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
