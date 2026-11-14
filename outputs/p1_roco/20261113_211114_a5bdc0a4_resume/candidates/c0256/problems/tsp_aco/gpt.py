import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = raw[valid]
    positive = values[values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
        robust_high = float(np.quantile(positive, 0.75))
    else:
        scale = 1.0
        robust_high = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    if not np.isfinite(robust_high) or robust_high <= 0.0:
        robust_high = scale

    epsilon = max(
        np.finfo(float).eps,
        1e-12 * max(scale, robust_high),
    )

    distances = np.where(valid, raw, np.inf)

    # Stable reciprocal-distance affinity with robust scale control.
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = 1.0 / (1.0 + distances[valid] / scale)

    # Outgoing and incoming rank signals preserve asymmetric evidence.
    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        count = int(np.sum(row_valid))
        if count == 0:
            continue

        order = np.argsort(
            np.where(row_valid, distances[i], np.inf),
            kind="stable",
        )
        rank = 0
        for j in order:
            if not row_valid[j]:
                continue
            outgoing_rank[i, j] = (
                1.0 if count == 1 else 1.0 - rank / (count - 1)
            )
            rank += 1

    for j in range(n):
        column_valid = valid[:, j]
        count = int(np.sum(column_valid))
        if count == 0:
            continue

        order = np.argsort(
            np.where(column_valid, distances[:, j], np.inf),
            kind="stable",
        )
        rank = 0
        for i in order:
            if not column_valid[i]:
                continue
            incoming_rank[i, j] = (
                1.0 if count == 1 else 1.0 - rank / (count - 1)
            )
            rank += 1

    rank_signal = 0.5 * (outgoing_rank + incoming_rank)

    # Conservative alternative-edge pressure at both endpoints.
    pressure = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_values = distances[i, valid[i]]
        if row_values.size <= 1:
            continue

        best_alternative = float(np.quantile(row_values, 0.50))
        local_scale = max(
            float(np.quantile(row_values, 0.75) - np.quantile(row_values, 0.25)),
            0.16 * scale,
            epsilon,
        )

        for j in np.flatnonzero(valid[i]):
            pressure[i, j] += np.clip(
                (best_alternative - distances[i, j]) / local_scale,
                0.0,
                1.0,
            )

    for j in range(n):
        column_values = distances[valid[:, j], j]
        if column_values.size <= 1:
            continue

        best_alternative = float(np.quantile(column_values, 0.50))
        local_scale = max(
            float(
                np.quantile(column_values, 0.75)
                - np.quantile(column_values, 0.25)
            ),
            0.16 * scale,
            epsilon,
        )

        for i in np.flatnonzero(valid[:, j]):
            pressure[i, j] += np.clip(
                (best_alternative - distances[i, j]) / local_scale,
                0.0,
                1.0,
            )

    pressure *= 0.5

    score = (
        0.68 * affinity
        + 0.27 * rank_signal
        + 0.05 * pressure
    )

    # Small exploration floor for every valid edge.
    score[valid] += 0.02
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    result = score / maximum
    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
