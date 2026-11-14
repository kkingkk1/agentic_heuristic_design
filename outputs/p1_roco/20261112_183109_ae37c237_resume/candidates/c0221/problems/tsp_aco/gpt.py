import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized local-structure-aware TSP edge priorities."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, distances, np.inf)

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    def endpoint_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)

        best = ordered[:, 0]
        second = ordered[:, 1] if n > 2 else best.copy()

        safe_best = np.where(
            np.isfinite(best),
            np.maximum(best, eps),
            global_scale,
        )
        safe_second = np.where(
            np.isfinite(second),
            np.maximum(second, eps),
            2.0 * safe_best,
        )

        scale = 0.5 * (safe_best + safe_second)
        scale = np.where(
            np.isfinite(scale) & (scale > 0.0),
            scale,
            global_scale,
        )
        scale = np.maximum(scale, eps)

        regret = np.divide(
            safe_second - safe_best,
            np.maximum(safe_second, eps),
        )
        regret = np.clip(regret, 0.0, 1.0)

        return scale, regret, safe_best

    out_scale, out_regret, out_best = endpoint_statistics(work)
    in_scale, in_regret, in_best = endpoint_statistics(work.T)

    pair_scale = np.sqrt(
        out_scale[:, None] * in_scale[None, :]
    )

    relative_distance = np.divide(
        distances,
        pair_scale,
        out=np.full_like(distances, np.inf),
        where=valid & (pair_scale > 0.0),
    )

    distance_affinity = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    out_ratio = np.divide(
        distances,
        out_scale[:, None],
        out=np.full_like(distances, np.inf),
        where=valid,
    )
    in_ratio = np.divide(
        distances,
        in_scale[None, :],
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

    column_order = np.argsort(work, axis=0, kind="stable")
    column_ranks = np.full((n, n), np.inf, dtype=float)
    column_ranks[
        column_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_affinity = np.exp(
        -0.5 * (row_ranks + column_ranks) / rank_scale
    )
    rank_affinity = np.nan_to_num(
        rank_affinity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    competitive_out = valid & (
        distances <= 1.75 * out_best[:, None]
    )
    competitive_in = valid & (
        distances <= 1.75 * in_best[None, :]
    )

    out_count = competitive_out.sum(axis=1).astype(float)
    in_count = competitive_in.sum(axis=0).astype(float)
    max_count = max(1.0, float(n - 1))

    out_scarcity = 1.0 - np.minimum(out_count / max_count, 1.0)
    in_scarcity = 1.0 - np.minimum(in_count / max_count, 1.0)

    scarcity = np.sqrt(
        out_scarcity[:, None] * in_scarcity[None, :]
    )
    scarcity_bonus = 1.0 + 0.10 * scarcity

    out_continuation = np.max(
        np.where(valid, out_quality, 0.0),
        axis=1,
    )
    in_continuation = np.max(
        np.where(valid, in_quality, 0.0),
        axis=0,
    )

    continuation = np.sqrt(
        out_continuation[None, :]
        * in_continuation[:, None]
    )
    continuation_bonus = 1.0 + 0.10 * np.clip(
        continuation,
        0.0,
        1.0,
    )

    regret_bonus = 1.0 + 0.30 * (
        out_regret[:, None] + in_regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.24 * np.maximum(relative_distance - 1.0, 0.0)
    )

    score = (
        distance_affinity
        * (
            0.60
            + 0.25 * endpoint_quality
            + 0.15 * rank_affinity
        )
        * regret_bonus
        * scarcity_bonus
        * continuation_bonus
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
    ).astype(float, copy=False)
