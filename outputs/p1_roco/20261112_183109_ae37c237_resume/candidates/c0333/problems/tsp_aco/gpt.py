import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust row-normalized edge-priority indicators for stochastic TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    eps = np.finfo(float).eps

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    reverse_valid = valid.T
    mutual_valid = valid & reverse_valid

    symmetric_input = np.array_equal(
        valid,
        reverse_valid,
    ) and np.allclose(
        distances[valid],
        distances.T[valid],
        rtol=1e-12,
        atol=1e-12,
    )

    work = np.where(valid, distances, np.inf)
    ordered = np.sort(work, axis=1)

    k = min(4, n - 1)
    nearest = ordered[:, :k]
    nearest_valid = np.isfinite(nearest)

    nearest_sum = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    )
    nearest_count = np.sum(nearest_valid, axis=1)

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.max(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    local_scale = np.divide(
        nearest_sum,
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    best = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, global_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        2.0 * safe_best,
    )

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.14 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    relative_distance = np.divide(
        distances,
        geometric_scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    distance_quality = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    row_ratio = np.divide(
        distances,
        row_scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        distances,
        col_scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 60.0))
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 60.0))
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(
        np.where(valid, distances, np.inf),
        axis=1,
        kind="stable",
    )
    row_rank_order = np.argsort(row_order, axis=1, kind="stable")
    row_ranks = np.where(valid, row_rank_order + 1.0, 0.0)

    col_order = np.argsort(
        np.where(valid, distances, np.inf),
        axis=0,
        kind="stable",
    )
    col_rank_order = np.argsort(col_order, axis=0, kind="stable")
    col_ranks = np.where(valid, col_rank_order + 1.0, 0.0)

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.07 * row_ranks)
        * (1.0 + 0.07 * col_ranks)
    )

    row_competition = np.sum(
        valid & (distances <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (distances <= safe_second[None, :]),
        axis=0,
        dtype=float,
    )

    scarcity = 1.0 + 0.08 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    support = proximity.sum(axis=1)
    positive_support = support[support > 0.0]
    support_scale = (
        float(np.median(positive_support))
        if positive_support.size
        else 1.0
    )
    support_scale = max(support_scale, eps)

    rarity = np.clip(
        np.sqrt(support_scale / np.maximum(support, eps)),
        0.80,
        1.45,
    )
    rarity_pair = 0.5 * (
        rarity[:, None] + rarity[None, :]
    )

    profile_norm = np.sqrt(np.sum(proximity * proximity, axis=1))
    profile_denominator = (
        profile_norm[:, None] * profile_norm[None, :]
    )

    overlap = np.divide(
        proximity @ proximity.T,
        profile_denominator,
        out=np.zeros((n, n), dtype=float),
        where=profile_denominator > 0.0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_signal = np.clip(
        (1.0 - overlap)
        * (1.0 - np.exp(-0.90 * np.maximum(relative_distance, 0.0))),
        0.0,
        1.0,
    )

    diversity_factor = np.clip(
        1.0
        + 0.06 * (rarity_pair - 1.0)
        + 0.045 * bridge_signal
        - 0.025 * overlap,
        0.90,
        1.10,
    )

    if np.any(mutual_valid):
        reverse_relative = np.divide(
            distances.T,
            geometric_scale,
            out=np.full_like(distances, np.inf, dtype=float),
            where=reverse_valid,
        )
        reverse_quality = np.exp(
            -np.clip(1.08 * reverse_relative, 0.0, 60.0)
        )
        bidirectional_quality = np.where(
            mutual_valid,
            np.sqrt(distance_quality * reverse_quality),
            distance_quality,
        )
    else:
        bidirectional_quality = distance_quality

    score = (
        bidirectional_quality
        * mutual_quality
        * rank_quality
        * scarcity
        * diversity_factor
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    if symmetric_input:
        score = 0.5 * (score + score.T)

    score = np.nan_to_num(
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
