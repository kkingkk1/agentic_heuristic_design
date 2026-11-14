import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed, row-normalized edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    work = np.where(valid, distances, np.inf)

    def endpoint_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)

        best = ordered[:, 0]
        second = ordered[:, 1] if n > 2 else best.copy()
        third = ordered[:, 2] if n > 3 else second.copy()

        safe_best = np.where(
            np.isfinite(best),
            np.maximum(best, eps),
            1.0,
        )
        safe_second = np.where(
            np.isfinite(second),
            np.maximum(second, eps),
            2.0 * safe_best,
        )
        safe_third = np.where(
            np.isfinite(third),
            np.maximum(third, eps),
            2.0 * safe_second,
        )

        local_scale = np.maximum(
            np.exp(
                (
                    np.log(safe_best)
                    + np.log(safe_second)
                    + np.log(safe_third)
                )
                / 3.0
            ),
            eps,
        )

        regret = np.clip(
            (safe_second - safe_best)
            / np.maximum(safe_second, eps),
            0.0,
            1.0,
        )

        competitive_limit = 1.8 * safe_best
        competitive_count = np.sum(
            np.isfinite(matrix)
            & (matrix <= competitive_limit[:, None]),
            axis=1,
        ).astype(float)

        return local_scale, regret, competitive_count

    out_scale, out_regret, out_count = endpoint_statistics(work)
    in_scale, in_regret, in_count = endpoint_statistics(work.T)

    pair_scale = np.sqrt(
        np.maximum(out_scale[:, None], eps)
        * np.maximum(in_scale[None, :], eps)
    )

    relative_distance = np.divide(
        distances,
        pair_scale,
        out=np.full_like(distances, np.inf),
        where=valid & (pair_scale > 0.0),
    )

    distance_quality = np.exp(
        -np.clip(1.02 * relative_distance, 0.0, 60.0)
    )

    out_ratio = np.divide(
        distances,
        np.maximum(out_scale[:, None], eps),
        out=np.full_like(distances, np.inf),
        where=valid,
    )
    in_ratio = np.divide(
        distances,
        np.maximum(in_scale[None, :], eps),
        out=np.full_like(distances, np.inf),
        where=valid,
    )

    out_quality = np.exp(-np.clip(out_ratio, 0.0, 60.0))
    in_quality = np.exp(-np.clip(in_ratio, 0.0, 60.0))
    endpoint_quality = np.sqrt(out_quality * in_quality)

    row_order = np.argsort(work, axis=1, kind="stable")
    row_ranks = np.full((n, n), np.inf, dtype=float)
    row_ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_ranks = np.full((n, n), np.inf, dtype=float)
    col_ranks[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, 0.95 * np.sqrt(float(n)))
    rank_quality = np.exp(
        -0.5 * (row_ranks + col_ranks) / rank_scale
    )
    rank_quality = np.nan_to_num(
        rank_quality,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    max_count = max(1.0, float(n - 1))
    out_scarcity = 1.0 - np.minimum(out_count / max_count, 1.0)
    in_scarcity = 1.0 - np.minimum(in_count / max_count, 1.0)

    scarcity_bonus = 1.0 + 0.10 * np.sqrt(
        out_scarcity[:, None] * in_scarcity[None, :]
    )

    continuation_out = np.max(
        np.where(valid, out_quality, 0.0),
        axis=1,
    )
    continuation_in = np.max(
        np.where(valid, in_quality, 0.0),
        axis=0,
    )
    continuation_potential = np.sqrt(
        continuation_out[:, None] * continuation_in[None, :]
    )
    continuation_bonus = 1.0 + 0.08 * np.clip(
        continuation_potential,
        0.0,
        1.0,
    )

    regret_bonus = 1.0 + 0.28 * (
        out_regret[:, None] + in_regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.28 * np.maximum(relative_distance - 1.0, 0.0)
    )

    preference = (
        np.power(np.maximum(distance_quality, 0.0), 0.58)
        * np.power(np.maximum(endpoint_quality, 0.0), 0.24)
        * np.power(np.maximum(rank_quality, 0.0), 0.18)
    )

    scores = (
        preference
        * regret_bonus
        * scarcity_bonus
        * continuation_bonus
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = scores.sum(axis=1, keepdims=True)

    return np.divide(
        scores,
        row_totals,
        out=np.zeros_like(scores),
        where=row_totals > 0.0,
    )
