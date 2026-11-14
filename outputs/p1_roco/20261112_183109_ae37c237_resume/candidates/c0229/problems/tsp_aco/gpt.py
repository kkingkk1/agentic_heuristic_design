import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized directional edge-priority indicators for TSP."""
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

    work = np.where(valid, distances, np.inf)
    eps = np.finfo(float).eps

    positive = distances[valid & (distances > 0.0)]
    if positive.size:
        fallback = float(np.median(positive))
    else:
        finite = distances[valid]
        fallback = float(np.median(finite)) if finite.size else 1.0

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    fallback = max(fallback, eps)

    def local_features(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)

        nearest = ordered[:, 0]
        if n >= 3:
            second = ordered[:, 1]
        else:
            second = nearest.copy()

        nearest = np.where(np.isfinite(nearest), nearest, fallback)
        second = np.where(np.isfinite(second), second, 2.0 * nearest)

        scale = 0.5 * (nearest + second)
        scale = np.where(
            np.isfinite(scale) & (scale > 0.0),
            scale,
            fallback,
        )
        scale = np.maximum(scale, eps)

        gap = np.divide(
            second - nearest,
            np.maximum(second, eps),
            out=np.zeros_like(second),
            where=np.isfinite(second),
        )
        gap = np.clip(gap, 0.0, 1.0)

        return scale, gap

    outgoing_scale, outgoing_gap = local_features(work)
    incoming_scale, incoming_gap = local_features(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    relative = np.divide(
        distances,
        pair_scale,
        out=np.full_like(distances, np.inf),
        where=valid,
    )

    outgoing_relative = np.divide(
        distances,
        outgoing_scale[:, None],
        out=np.full_like(distances, np.inf),
        where=valid,
    )
    incoming_relative = np.divide(
        distances,
        incoming_scale[None, :],
        out=np.full_like(distances, np.inf),
        where=valid,
    )

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.full((n, n), np.inf, dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_bonus = np.exp(
        -0.5 * (row_rank + col_rank) / rank_scale
    )
    rank_bonus = np.nan_to_num(
        rank_bonus,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    relative_clip = np.clip(relative, 0.0, 60.0)
    outgoing_clip = np.clip(outgoing_relative, 0.0, 60.0)
    incoming_clip = np.clip(incoming_relative, 0.0, 60.0)

    log_scores = (
        -1.35 * relative_clip
        -0.20 * np.maximum(relative_clip - 1.0, 0.0)
        -0.20 * outgoing_clip
        -0.20 * incoming_clip
        +0.12 * rank_bonus
        +np.log1p(
            0.20 * (
                outgoing_gap[:, None]
                + incoming_gap[None, :]
            )
        )
    )

    log_scores[~valid] = -np.inf
    log_scores[~np.isfinite(log_scores)] = -np.inf

    row_max = np.max(log_scores, axis=1, keepdims=True)
    stable = np.where(
        np.isfinite(row_max),
        np.exp(log_scores - row_max),
        0.0,
    )
    stable[~valid] = 0.0

    row_totals = stable.sum(axis=1, keepdims=True)
    result = np.divide(
        stable,
        row_totals,
        out=np.zeros_like(stable),
        where=row_totals > 0.0,
    )

    return result.astype(float, copy=False)
