import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust row-normalized directed edge priorities for stochastic TSP sampling."""
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

    eps = np.finfo(float).eps
    transformed = np.where(valid, np.log1p(d), np.nan)

    finite_values = transformed[valid]
    global_center = float(np.median(finite_values))
    global_deviation = np.abs(finite_values - global_center)
    global_scale = float(np.median(global_deviation))

    if not np.isfinite(global_center):
        global_center = 0.0
    if not np.isfinite(global_scale) or global_scale < eps:
        spread = float(np.std(finite_values))
        global_scale = spread if np.isfinite(spread) and spread > eps else 1.0
    global_scale = max(global_scale, eps)

    row_center = np.full(n, global_center, dtype=float)
    row_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = transformed[i, valid[i]]
        if values.size == 0:
            continue

        center = float(np.median(values))
        deviation = float(np.median(np.abs(values - center)))

        if not np.isfinite(center):
            center = global_center
        if not np.isfinite(deviation) or deviation < eps:
            deviation = global_scale

        row_center[i] = center
        row_scale[i] = max(deviation, eps)

    column_center = np.full(n, global_center, dtype=float)
    column_scale = np.full(n, global_scale, dtype=float)

    for j in range(n):
        values = transformed[valid[:, j], j]
        if values.size == 0:
            continue

        center = float(np.median(values))
        deviation = float(np.median(np.abs(values - center)))

        if not np.isfinite(center):
            center = global_center
        if not np.isfinite(deviation) or deviation < eps:
            deviation = global_scale

        column_center[j] = center
        column_scale[j] = max(deviation, eps)

    row_residual = np.divide(
        transformed - row_center[:, None],
        row_scale[:, None],
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    column_residual = np.divide(
        transformed - column_center[None, :],
        column_scale[None, :],
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )

    global_residual = (transformed - global_center) / global_scale

    outgoing_count = valid.sum(axis=1).astype(float)
    incoming_count = valid.sum(axis=0).astype(float)

    outgoing_support = np.log1p(outgoing_count)
    incoming_support = np.log1p(incoming_count)

    outgoing_support /= max(float(np.max(outgoing_support)), 1.0)
    incoming_support /= max(float(np.max(incoming_support)), 1.0)

    source_term = outgoing_support[:, None]
    destination_term = incoming_support[None, :]

    logits = (
        -0.78 * row_residual
        -0.42 * column_residual
        -0.18 * global_residual
        +0.20 * destination_term
        +0.08 * source_term
    )

    logits = np.where(valid, np.clip(logits, -60.0, 60.0), -np.inf)

    row_max = np.max(logits, axis=1, keepdims=True)
    stable_logits = np.where(
        valid,
        logits - np.where(np.isfinite(row_max), row_max, 0.0),
        -np.inf,
    )

    weights = np.where(valid, np.exp(stable_logits), 0.0)
    weights = np.nan_to_num(
        weights,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    totals = weights.sum(axis=1, keepdims=True)

    return np.divide(
        weights,
        totals,
        out=result,
        where=totals > 0.0,
    )
