import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    k = min(4, n - 1)
    neighbors = sorted_rows[:, :k]
    finite_neighbors = np.isfinite(neighbors)
    counts = finite_neighbors.sum(axis=1)

    safe_neighbors = np.where(finite_neighbors, neighbors, 0.0)
    local_scale = np.divide(
        safe_neighbors.sum(axis=1),
        np.maximum(counts, 1),
        out=np.ones(n, dtype=float),
        where=counts > 0,
    )

    positive_scales = local_scale[np.isfinite(local_scale) & (local_scale > 0.0)]
    fallback_scale = (
        float(np.median(positive_scales))
        if positive_scales.size
        else 1.0
    )
    local_scale = np.maximum(
        np.where(np.isfinite(local_scale), local_scale, fallback_scale),
        np.finfo(float).eps,
    )

    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.16 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    finite_distances = d[valid]
    distance_median = float(np.median(finite_distances))
    distance_mean = float(np.mean(finite_distances))
    distance_std = float(np.std(finite_distances))
    dispersion = distance_std / max(distance_mean, np.finfo(float).eps)

    temperature = 0.90 + 0.45 * np.clip(
        dispersion / (1.0 + dispersion),
        0.0,
        1.0,
    )

    distance_quality = np.exp(
        -np.clip(temperature * normalized_distance, 0.0, 50.0)
    )

    row_ratio = np.divide(
        d,
        row_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 50.0))
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 50.0))
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_ranks = np.zeros_like(d, dtype=float)
    col_ranks = np.zeros_like(d, dtype=float)

    row_ranks[valid] = (
        np.argsort(
            np.argsort(np.where(valid, d, np.inf), axis=1),
            axis=1,
        )[valid]
        + 1.0
    )
    col_ranks[valid] = (
        np.argsort(
            np.argsort(np.where(valid, d, np.inf), axis=0),
            axis=0,
        )[valid]
        + 1.0
    )

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.10 * row_ranks)
        * (1.0 + 0.10 * col_ranks)
    )

    row_competition = np.sum(
        valid & (d <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (d <= safe_second[None, :]),
        axis=0,
        dtype=float,
    )

    scarcity = 1.0 + 0.06 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    scale_contrast = np.abs(
        np.log(row_scale / np.maximum(col_scale, np.finfo(float).eps))
    )
    residual_diversity = 1.0 + 0.08 * np.tanh(scale_contrast)

    bridge_excess = np.maximum(
        np.maximum(row_ratio, col_ratio) - 1.0,
        0.0,
    )
    bridge_penalty = 1.0 / (
        1.0 + 0.10 * np.minimum(bridge_excess, 8.0) ** 2
    )

    score = (
        distance_quality
        * mutual_quality
        * rank_quality
        * scarcity
        * residual_diversity
        * bridge_penalty
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    base = np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )

    valid_counts = valid.sum(axis=1, keepdims=True).astype(float)
    uniform = np.divide(
        valid.astype(float),
        np.maximum(valid_counts, 1.0),
    )

    concentration = np.max(base, axis=1, keepdims=True)
    mix = 0.04 + 0.16 * np.clip(
        (concentration - 1.0 / np.maximum(valid_counts, 1.0))
        / np.maximum(
            1.0 - 1.0 / np.maximum(valid_counts, 1.0),
            np.finfo(float).eps,
        ),
        0.0,
        1.0,
    )

    result = (1.0 - mix) * base + mix * uniform
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    final_totals = result.sum(axis=1, keepdims=True)
    return np.divide(
        result,
        final_totals,
        out=np.zeros_like(result),
        where=final_totals > 0.0,
    )
