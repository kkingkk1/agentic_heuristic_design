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

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    work = np.where(valid, distances, np.inf)

    def local_profile(matrix: np.ndarray):
        degree = np.sum(np.isfinite(matrix), axis=1)
        k = min(3, max(1, n - 1))

        ordered = np.sort(matrix, axis=1)
        kth = ordered[:, k - 1]

        nearest = ordered[:, 0]
        second = ordered[:, min(1, n - 2)]

        nearest = np.where(np.isfinite(nearest), nearest, scale)
        second = np.where(np.isfinite(second), second, nearest + scale)
        kth = np.where(np.isfinite(kth), kth, 0.5 * (nearest + second))

        local = np.maximum(0.5 * (nearest + kth), np.finfo(float).eps)

        regret = np.divide(
            second - nearest,
            np.maximum(second, np.finfo(float).eps),
        )
        regret = np.clip(regret, 0.0, 1.0)

        return local, regret, degree

    outgoing_scale, outgoing_regret, outgoing_degree = local_profile(work)
    incoming_scale, incoming_regret, incoming_degree = local_profile(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )

    normalized = np.divide(
        distances,
        pair_scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    log_priority = -np.log1p(np.maximum(normalized, 0.0))

    outgoing_ratio = np.divide(
        distances,
        outgoing_scale[:, None],
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )
    incoming_ratio = np.divide(
        distances,
        incoming_scale[None, :],
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    local_compatibility = -0.5 * (
        np.log1p(np.maximum(outgoing_ratio, 0.0))
        + np.log1p(np.maximum(incoming_ratio, 0.0))
    )

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    candidate_width = min(5, max(1, n - 1))
    mutual_candidate = (
        (row_rank < candidate_width)
        & (col_rank < candidate_width)
        & valid
    )

    regret_bonus = 0.12 * (
        outgoing_regret[:, None] + incoming_regret[None, :]
    )

    sparsity_bonus = 0.05 * (
        np.divide(
            1.0,
            np.maximum(outgoing_degree[:, None], 1),
        )
        + np.divide(
            1.0,
            np.maximum(incoming_degree[None, :], 1),
        )
    )

    log_score = (
        1.20 * log_priority
        + 0.35 * local_compatibility
        + regret_bonus
        + sparsity_bonus
        + 0.18 * mutual_candidate.astype(float)
    )

    log_score[~valid] = -np.inf
    row_max = np.max(log_score, axis=1, keepdims=True)

    stable = np.exp(
        np.clip(log_score - row_max, -60.0, 0.0),
        where=np.isfinite(log_score - row_max),
        out=np.zeros_like(log_score),
    )
    stable[~valid] = 0.0
    np.fill_diagonal(stable, 0.0)

    totals = stable.sum(axis=1, keepdims=True)

    return np.divide(
        stable,
        totals,
        out=np.zeros_like(stable),
        where=totals > 0.0,
    ).astype(float, copy=False)
