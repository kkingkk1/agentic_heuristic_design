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

    def neighborhood_statistics(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ordered = np.sort(matrix, axis=1)
        k = min(7, matrix.shape[1] - 1)
        neighborhood = ordered[:, :k]

        finite = np.isfinite(neighborhood)
        counts = finite.sum(axis=1)
        sums = np.where(finite, neighborhood, 0.0).sum(axis=1)

        scale = np.divide(
            sums,
            counts,
            out=np.ones(matrix.shape[0], dtype=float),
            where=counts > 0,
        )
        scale = np.maximum(scale, eps)

        best = ordered[:, 0]
        second = ordered[:, 1] if matrix.shape[1] > 2 else best * 2.0

        best_safe = np.where(np.isfinite(best), best, 1.0)
        second_safe = np.where(
            np.isfinite(second),
            second,
            2.0 * best_safe,
        )

        regret = np.divide(
            second_safe - best_safe,
            np.maximum(second_safe, eps),
        )
        regret = np.clip(regret, 0.0, 1.0)

        return scale, regret, ordered

    out_scale, out_regret, out_ordered = neighborhood_statistics(work)
    in_scale, in_regret, in_ordered = neighborhood_statistics(work.T)

    pair_scale = np.sqrt(
        np.maximum(out_scale[:, None], eps)
        * np.maximum(in_scale[None, :], eps)
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )
    relative_distance = np.maximum(relative_distance, 0.0)

    distance_quality = np.exp(
        -np.clip(0.95 * relative_distance, 0.0, 60.0)
    )

    outgoing_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                out_scale[:, None],
                out=np.full_like(d, np.inf),
                where=valid,
            ),
            0.0,
            60.0,
        )
    )
    incoming_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                in_scale[None, :],
                out=np.full_like(d, np.inf),
                where=valid,
            ),
            0.0,
            60.0,
        )
    )

    outgoing_quality *= valid
    incoming_quality *= valid

    mutual_quality = np.sqrt(outgoing_quality * incoming_quality)

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_quality = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )
    rank_quality *= valid

    regret_bonus = 1.0 + 0.40 * (
        out_regret[:, None] + in_regret[None, :]
    )

    destination_options = np.max(outgoing_quality, axis=1)
    source_options = np.max(incoming_quality, axis=0)

    continuation = np.sqrt(
        destination_options[None, :]
        * source_options[:, None]
    )
    continuation_bonus = 1.0 + 0.12 * continuation

    nearest_outgoing = np.argmin(work, axis=1)
    nearest_incoming = np.argmin(work, axis=0)

    mutual_nearest = (
        (nearest_outgoing[:, None] == np.arange(n)[None, :])
        & (nearest_incoming[None, :] == np.arange(n)[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.18, 1.0)

    score = (
        distance_quality
        * (0.55 + 0.45 * mutual_quality)
        * (0.68 + 0.32 * rank_quality)
        * regret_bonus
        * continuation_bonus
        * mutual_bonus
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
