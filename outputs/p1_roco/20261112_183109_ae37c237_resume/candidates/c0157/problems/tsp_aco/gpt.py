import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return local-structure-aware edge-priority indicators for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else float(np.max(finite_values))
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)

    def local_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)
        best = ordered[:, 0]

        if n > 2:
            second = ordered[:, 1]
        else:
            second = best.copy()

        safe_best = np.where(np.isfinite(best), best, global_scale)
        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * safe_best,
        )

        scale = 0.5 * (safe_best + safe_second)
        scale = np.where(
            np.isfinite(scale) & (scale > 0.0),
            scale,
            global_scale,
        )
        scale = np.maximum(scale, eps)

        regret = (safe_second - safe_best) / np.maximum(safe_second, eps)
        regret = np.clip(regret, 0.0, 1.0)

        order = np.argsort(matrix, axis=1, kind="stable")
        ranks = np.empty((n, n), dtype=float)
        ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

        nearest = order[:, 0]
        has_neighbor = np.isfinite(best)

        return scale, regret, ranks, nearest, has_neighbor

    out_scale, out_regret, out_rank, out_nearest, out_has_neighbor = (
        local_statistics(work)
    )
    in_scale, in_regret, in_rank, in_nearest, in_has_neighbor = (
        local_statistics(work.T)
    )

    pair_scale = np.sqrt(
        out_scale[:, None] * in_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )
    distance_quality = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    out_ratio = np.divide(
        d,
        out_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    in_ratio = np.divide(
        d,
        in_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    out_quality = np.exp(-np.clip(out_ratio, 0.0, 60.0))
    in_quality = np.exp(-np.clip(in_ratio, 0.0, 60.0))
    mutual_quality = np.sqrt(out_quality * in_quality)

    rank_quality = np.exp(
        -0.5
        * (
            out_rank / max(1.0, np.sqrt(float(n)))
            + in_rank / max(1.0, np.sqrt(float(n)))
        )
    )

    regret_bonus = 1.0 + 0.30 * (
        out_regret[:, None] + in_regret[None, :]
    )

    mutual_nearest = (
        out_has_neighbor[:, None]
        & in_has_neighbor[None, :]
        & (out_nearest[:, None] == np.arange(n)[None, :])
        & (in_nearest[None, :] == np.arange(n)[:, None])
    )
    nearest_bonus = np.where(mutual_nearest, 1.20, 1.0)

    continuation = np.sqrt(
        np.max(np.where(valid, out_quality, 0.0), axis=1)[:, None]
        * np.max(np.where(valid, in_quality, 0.0), axis=0)[None, :]
    )
    continuation_bonus = 1.0 + 0.08 * continuation

    score = (
        distance_quality
        * (0.55 + 0.45 * mutual_quality)
        * (0.75 + 0.25 * rank_quality)
        * regret_bonus
        * nearest_bonus
        * continuation_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    is_symmetric = np.allclose(
        d,
        d.T,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    if is_symmetric:
        score = 0.5 * (score + score.T)
        maximum = np.max(score)
        if np.isfinite(maximum) and maximum > 0.0:
            score /= maximum
        return np.nan_to_num(
            score,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
