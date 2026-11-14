import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directional, scale-invariant edge desirability indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    valid = (
        np.isfinite(distances)
        & (distances > 0)
    )
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    positive_values = distances[valid]
    global_scale = float(np.median(positive_values))
    global_scale = max(global_scale, np.finfo(float).eps)

    row_scales = np.full(n, global_scale, dtype=float)
    col_scales = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = distances[i, valid[i]]
        if values.size:
            row_scales[i] = max(float(np.median(values)), np.finfo(float).eps)

    for j in range(n):
        values = distances[valid[:, j], j]
        if values.size:
            col_scales[j] = max(float(np.median(values)), np.finfo(float).eps)

    row_log_scales = np.log(row_scales)
    col_log_scales = np.log(col_scales)
    global_log_scale = np.log(global_scale)

    row_spread = np.zeros(n, dtype=float)
    for i in range(n):
        values = distances[i, valid[i]]
        if values.size >= 2:
            row_spread[i] = np.std(np.log(np.maximum(values, np.finfo(float).eps)))

    continuation = np.zeros(n, dtype=float)
    for j in range(n):
        outgoing = distances[j, valid[j]]
        if outgoing.size:
            normalized = outgoing / max(row_scales[j], np.finfo(float).eps)
            continuation[j] = 1.0 / max(float(np.min(normalized)), 1e-12)

    pressure = np.zeros(n, dtype=float)
    for j in range(n):
        incoming = distances[valid[:, j], j]
        if incoming.size:
            normalized = incoming / max(col_scales[j], np.finfo(float).eps)
            pressure[j] = float(np.mean(np.exp(-np.minimum(normalized, 50.0))))

    pressure = pressure / max(float(np.mean(pressure[pressure > 0])), 1e-12)
    pressure_factor = 1.0 / np.sqrt(0.5 + pressure)

    for i in range(n):
        js = np.flatnonzero(valid[i])
        if js.size == 0:
            continue

        d = distances[i, js]
        log_d = np.log(np.maximum(d, np.finfo(float).eps))

        source_relative = d / np.sqrt(row_scales[i] * col_scales[js])
        global_relative = d / global_scale

        local_term = np.exp(-np.minimum(source_relative, 50.0))
        global_term = 1.0 / (1.0 + global_relative)

        scale_mismatch = np.abs(
            row_log_scales[i] - col_log_scales[js]
        )
        scale_compatibility = np.exp(-0.25 * np.minimum(scale_mismatch, 50.0))

        continuation_term = np.sqrt(
            np.maximum(continuation[js], 1e-12)
            / max(1.0 / max(row_scales[i], 1e-12), 1e-12)
        )
        continuation_term = 1.0 / (1.0 + continuation_term)

        spread_factor = 1.0 + 0.15 * np.tanh(row_spread[i])
        diversity_term = pressure_factor[js] * spread_factor

        edge_score = (
            (0.60 * local_term + 0.40 * global_term)
            * scale_compatibility
            * (0.65 + 0.35 * continuation_term)
            * diversity_term
        )

        edge_score *= np.exp(
            -0.05 * np.abs(log_d - np.median(log_d))
        )

        edge_score = np.nan_to_num(
            edge_score,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        maximum = float(np.max(edge_score))
        if maximum > 0:
            scores[i, js] = edge_score / maximum

    np.fill_diagonal(scores, 0.0)
    return scores.astype(float, copy=False)
