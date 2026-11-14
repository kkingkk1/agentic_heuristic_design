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

    finite_values = d[valid]
    global_scale = float(np.median(finite_values))
    global_scale = max(global_scale, np.finfo(float).eps)

    work = np.where(valid, d, np.inf)
    k = min(4, n - 1)
    nearest = np.sort(work, axis=1)[:, :k]

    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)
    safe_nearest = np.where(finite_nearest, nearest, 0.0)

    local_scale = np.divide(
        safe_nearest.sum(axis=1),
        np.maximum(counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    best = nearest[:, 0]
    second = nearest[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )

    # Stronger decay is used where the nearest edge is clearly preferable.
    decay = 0.92 + 0.42 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]

    geometric_scale = np.sqrt(row_scale * col_scale)
    arithmetic_scale = 0.5 * (row_scale + col_scale)
    blended_scale = 0.68 * geometric_scale + 0.32 * arithmetic_scale
    blended_scale = np.maximum(blended_scale, np.finfo(float).eps)

    normalized_distance = np.divide(
        d,
        blended_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
        / decay[:, None]
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

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 50.0) / 1.10)
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 50.0) / 1.10)

    # Mild mutual-neighborhood support avoids overpowering distance quality.
    mutual_support = np.sqrt(row_quality * col_quality)
    mutual_factor = 0.82 + 0.18 * mutual_support

    # Mild endpoint urgency helps retain edges incident to constrained nodes.
    urgency = 1.0 + 0.20 * regret
    endpoint_factor = np.sqrt(urgency[:, None] * urgency[None, :])

    score = distance_score * mutual_factor * endpoint_factor
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    base = np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )

    # Adaptive exploration floor: larger only when the row is overly concentrated.
    squared_totals = np.sum(base * base, axis=1)
    concentration = np.clip(
        (squared_totals - 1.0 / max(n - 1, 1))
        / max(1.0 - 1.0 / max(n - 1, 1), np.finfo(float).eps),
        0.0,
        1.0,
    )

    floor = 0.035 + 0.055 * concentration
    valid_counts = valid.sum(axis=1).astype(float)

    uniform_valid = np.divide(
        valid.astype(float),
        valid_counts[:, None],
        out=np.zeros_like(score),
        where=valid_counts[:, None] > 0.0,
    )

    mixed = (1.0 - floor[:, None]) * base + floor[:, None] * uniform_valid
    mixed[~valid] = 0.0
    np.fill_diagonal(mixed, 0.0)

    final_totals = mixed.sum(axis=1, keepdims=True)
    return np.divide(
        mixed,
        final_totals,
        out=result,
        where=final_totals > 0.0,
    )
