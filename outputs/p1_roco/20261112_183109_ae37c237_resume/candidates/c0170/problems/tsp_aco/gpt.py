import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    if n == 2:
        nearest = work.copy()
        second = work.copy()
    else:
        two_smallest = np.partition(work, kth=1, axis=1)[:, :2]
        nearest = two_smallest[:, 0]
        second = two_smallest[:, 1]

    def endpoint_stats(matrix: np.ndarray):
        if matrix.shape[0] == 2:
            best = matrix[:, 0]
            second_best = matrix[:, 0]
        else:
            smallest = np.partition(matrix, kth=1, axis=1)[:, :2]
            best = smallest[:, 0]
            second_best = smallest[:, 1]

        finite_best = np.isfinite(best)
        safe_best = np.where(finite_best, best, 1.0)
        safe_second = np.where(
            np.isfinite(second_best),
            second_best,
            2.0 * np.maximum(safe_best, eps),
        )

        scale = np.maximum(0.5 * (safe_best + safe_second), eps)
        regret = np.clip(
            (safe_second - safe_best)
            / np.maximum(safe_second, eps),
            0.0,
            1.0,
        )
        return scale, regret

    outgoing_scale, outgoing_regret = endpoint_stats(work)
    incoming_scale, incoming_regret = endpoint_stats(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )

    outgoing_ratio = np.divide(
        d,
        outgoing_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    incoming_ratio = np.divide(
        d,
        incoming_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    relative_distance = np.nan_to_num(
        relative_distance,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    outgoing_ratio = np.nan_to_num(
        outgoing_ratio,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    incoming_ratio = np.nan_to_num(
        incoming_ratio,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )

    local_quality = np.exp(
        -0.65 * np.clip(relative_distance, 0.0, 60.0)
        -0.18 * np.clip(outgoing_ratio, 0.0, 60.0)
        -0.18 * np.clip(incoming_ratio, 0.0, 60.0)
    )

    outgoing_best = np.max(
        np.where(valid, np.exp(-np.clip(outgoing_ratio, 0.0, 60.0)), 0.0),
        axis=1,
    )
    incoming_best = np.max(
        np.where(valid, np.exp(-np.clip(incoming_ratio, 0.0, 60.0)), 0.0),
        axis=0,
    )

    continuation = np.sqrt(
        outgoing_best[:, None] * incoming_best[None, :]
    )

    regret_bonus = 1.0 + 0.35 * (
        outgoing_regret[:, None] + incoming_regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.22 * np.maximum(relative_distance - 1.0, 0.0)
    )

    score = (
        local_quality
        * (0.78 + 0.14 * continuation)
        * regret_bonus
        * long_edge_penalty
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0.0,
    )
