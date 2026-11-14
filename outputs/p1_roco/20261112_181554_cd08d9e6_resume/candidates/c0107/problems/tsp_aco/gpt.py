import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized stochastic edge-priority indicators for TSP."""
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
    work = np.where(valid, d, np.inf)

    def local_statistics(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ordered = np.sort(matrix, axis=1)
        best = ordered[:, 0]

        if matrix.shape[1] > 2:
            second = ordered[:, 1]
        else:
            second = best * 2.0

        finite_best = np.isfinite(best)
        safe_best = np.where(finite_best, best, 1.0)
        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * safe_best,
        )

        scale = np.where(
            finite_best,
            0.5 * (safe_best + safe_second),
            1.0,
        )
        scale = np.maximum(scale, eps)

        regret = (safe_second - safe_best) / np.maximum(safe_second, eps)
        regret = np.clip(regret, 0.0, 1.0)

        return scale, regret

    outgoing_scale, outgoing_regret = local_statistics(work)
    incoming_scale, incoming_regret = local_statistics(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )
    distance_quality = np.exp(-np.clip(relative_distance, 0.0, 60.0))

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

    outgoing_quality = np.exp(-np.clip(outgoing_ratio, 0.0, 60.0))
    incoming_quality = np.exp(-np.clip(incoming_ratio, 0.0, 60.0))

    mutual_quality = np.sqrt(outgoing_quality * incoming_quality)

    ranked = np.argsort(
        np.where(valid, d, np.inf),
        axis=1,
        kind="stable",
    )
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], ranked] = np.arange(n)

    incoming_ranked = np.argsort(
        np.where(valid, d, np.inf),
        axis=0,
        kind="stable",
    )
    col_rank = np.empty((n, n), dtype=float)
    col_rank[incoming_ranked, np.arange(n)[None, :]] = np.arange(n)

    rank_quality = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    regret_bonus = 1.0 + 0.45 * (
        outgoing_regret[:, None] + incoming_regret[None, :]
    )

    destination_options = np.max(
        np.where(valid, outgoing_quality, 0.0),
        axis=1,
    )
    source_options = np.max(
        np.where(valid, incoming_quality, 0.0),
        axis=0,
    )

    continuation = np.sqrt(
        destination_options[None, :] * source_options[:, None]
    )
    continuation_bonus = 1.0 + 0.10 * continuation

    score = (
        distance_quality
        * (0.55 + 0.45 * mutual_quality)
        * (0.65 + 0.35 * rank_quality)
        * regret_bonus
        * continuation_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
