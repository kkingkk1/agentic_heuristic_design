import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = raw[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        q25, q50, q75 = np.quantile(
            positive_values,
            [0.25, 0.50, 0.75],
        )
        spread = float(q75 - q25)
        scale = max(float(q50), 0.5 * spread, np.finfo(float).eps)
    else:
        scale = 1.0

    distances = np.where(valid, np.maximum(raw, 0.0), np.inf)

    # Stable reciprocal-distance affinity.
    base = np.zeros((n, n), dtype=float)
    base[valid] = 1.0 / (1.0 + distances[valid] / scale)

    # Directional rank evidence from both endpoints.
    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(distances[i, indices], kind="stable")]
            denom = max(order.size - 1, 1)
            outgoing_rank[i, order] = 1.0 - (
                np.arange(order.size, dtype=float) / denom
            )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[
                np.argsort(distances[indices, j], kind="stable")
            ]
            denom = max(order.size - 1, 1)
            incoming_rank[order, j] = 1.0 - (
                np.arange(order.size, dtype=float) / denom
            )

    bidirectional_rank = np.sqrt(outgoing_rank * incoming_rank)

    # Conservative alternative-edge pressure.
    alternative_pressure = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_indices = np.flatnonzero(valid[i])
        if row_indices.size >= 2:
            row_values = np.sort(distances[i, row_indices])
            second_best = float(row_values[1])
        elif row_indices.size == 1:
            second_best = float(distances[i, row_indices[0]]) + scale
        else:
            continue

        for j in row_indices:
            gap = max(second_best - distances[i, j], 0.0)
            alternative_pressure[i, j] = 1.0 - np.exp(
                -gap / max(0.75 * scale, np.finfo(float).eps)
            )

    for j in range(n):
        column_indices = np.flatnonzero(valid[:, j])
        if column_indices.size >= 2:
            column_values = np.sort(distances[column_indices, j])
            second_best = float(column_values[1])
        elif column_indices.size == 1:
            second_best = float(distances[column_indices[0], j]) + scale
        else:
            continue

        for i in column_indices:
            gap = max(second_best - distances[i, j], 0.0)
            value = 1.0 - np.exp(
                -gap / max(0.75 * scale, np.finfo(float).eps)
            )
            alternative_pressure[i, j] = max(
                alternative_pressure[i, j],
                value,
            )

    # Small two-step lookahead correction for edges that are hard to bypass.
    best_two_step = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        left = distances[:, k][:, None]
        right = distances[k, :][None, :]
        best_two_step = np.minimum(best_two_step, left + right)

    bypass_gain = np.zeros((n, n), dtype=float)
    bypass_valid = valid & np.isfinite(best_two_step)
    bypass_gain[bypass_valid] = np.maximum(
        best_two_step[bypass_valid] - distances[bypass_valid],
        0.0,
    )
    bypass_signal = np.zeros((n, n), dtype=float)
    bypass_signal[bypass_valid] = 1.0 - np.exp(
        -bypass_gain[bypass_valid] / max(scale, np.finfo(float).eps)
    )

    # Keep distance affinity dominant and local corrections modest.
    score = (
        0.58 * base
        + 0.22 * bidirectional_rank
        + 0.14 * alternative_pressure
        + 0.06 * bypass_signal
    )
    score[~valid] = 0.0

    # A small exploration floor applies only to valid edges.
    floor = 0.015
    score[valid] = floor + (1.0 - floor) * score[valid]

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
