import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative row-normalized edge priorities for stochastic TSP sampling."""
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
    nearest = sorted_rows[:, :k]
    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)

    local_scale = np.divide(
        np.where(finite_nearest, nearest, 0.0).sum(axis=1),
        np.maximum(counts, 1),
        out=np.ones(n, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(np.abs(safe_second), np.finfo(float).eps),
        0.0,
        1.0,
    )

    # Mild urgency: useful for isolated nodes without making them dominant.
    urgency = 1.0 + 0.12 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    distance_quality = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0) / 1.00
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

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 50.0) / 1.00)
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 50.0) / 1.00)
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(np.where(valid, d, np.inf), axis=1)
    row_ranks = np.empty_like(row_order, dtype=float)
    row_ranks[np.arange(n)[:, None], row_order] = (
        np.arange(n, dtype=float)[None, :] + 1.0
    )

    col_order = np.argsort(np.where(valid, d, np.inf), axis=0)
    col_ranks = np.empty_like(col_order, dtype=float)
    col_ranks[col_order, np.arange(n)[None, :]] = (
        np.arange(n, dtype=float)[:, None] + 1.0
    )

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.08 * row_ranks) * (1.0 + 0.08 * col_ranks)
    )

    # Candidate availability provides a soft feasibility proxy.
    row_threshold = safe_second[:, None]
    col_threshold = safe_second[None, :]

    row_options = np.sum(
        valid & (d <= row_threshold),
        axis=1,
        dtype=float,
    )
    col_options = np.sum(
        valid & (d <= col_threshold),
        axis=0,
        dtype=float,
    )

    endpoint_options = np.sqrt(
        np.maximum(row_options[:, None], 1.0)
        * np.maximum(col_options[None, :], 1.0)
    )

    # Slightly favor scarce endpoints, while discouraging extreme concentration.
    scarcity = 1.0 + 0.055 / np.sqrt(endpoint_options)
    diversity = 1.0 - 0.06 / np.sqrt(endpoint_options)

    score = (
        distance_quality
        * mutual_quality
        * rank_quality
        * scarcity
        * diversity
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    # Keep a small exploration floor among valid edges.
    positive = score > 0.0
    row_max = np.max(np.where(positive, score, 0.0), axis=1, keepdims=True)
    floor = 0.015 * row_max
    score = np.where(valid, np.maximum(score, floor), 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
