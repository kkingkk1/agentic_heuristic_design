import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed, row-normalized TSP edge-priority indicators."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps

    positive = distances[valid & (distances > 0.0)]
    if positive.size:
        base_scale = float(np.median(positive))
    else:
        finite_values = distances[valid]
        base_scale = float(np.mean(finite_values)) if finite_values.size else 1.0

    if not np.isfinite(base_scale) or base_scale <= 0.0:
        base_scale = 1.0
    base_scale = max(base_scale, eps)

    safe_distances = np.where(valid, distances, 0.0)
    row_counts = valid.sum(axis=1).astype(float)
    col_counts = valid.sum(axis=0).astype(float)

    row_mean = np.divide(
        safe_distances.sum(axis=1),
        row_counts,
        out=np.full(n, base_scale, dtype=float),
        where=row_counts > 0.0,
    )
    col_mean = np.divide(
        safe_distances.sum(axis=0),
        col_counts,
        out=np.full(n, base_scale, dtype=float),
        where=col_counts > 0.0,
    )

    row_deviation = np.zeros(n, dtype=float)
    col_deviation = np.zeros(n, dtype=float)

    row_deviation = np.divide(
        np.where(valid, np.abs(distances - row_mean[:, None]), 0.0).sum(axis=1),
        row_counts,
        out=np.zeros(n, dtype=float),
        where=row_counts > 0.0,
    )
    col_deviation = np.divide(
        np.where(valid, np.abs(distances - col_mean[None, :]), 0.0).sum(axis=0),
        col_counts,
        out=np.zeros(n, dtype=float),
        where=col_counts > 0.0,
    )

    row_scale = np.maximum(row_deviation, 0.20 * np.maximum(row_mean, base_scale * eps))
    col_scale = np.maximum(col_deviation, 0.20 * np.maximum(col_mean, base_scale * eps))

    log_distances = np.log1p(np.maximum(distances, 0.0) / base_scale)

    row_log_mean = np.divide(
        np.where(valid, log_distances, 0.0).sum(axis=1),
        row_counts,
        out=np.full(n, np.log1p(base_scale / base_scale), dtype=float),
        where=row_counts > 0.0,
    )
    col_log_mean = np.divide(
        np.where(valid, log_distances, 0.0).sum(axis=0),
        col_counts,
        out=np.full(n, np.log1p(base_scale / base_scale), dtype=float),
        where=col_counts > 0.0,
    )

    row_log_scale = np.maximum(
        np.divide(
            np.where(
                valid,
                np.abs(log_distances - row_log_mean[:, None]),
                0.0,
            ).sum(axis=1),
            row_counts,
            out=np.ones(n, dtype=float),
            where=row_counts > 0.0,
        ),
        0.15,
    )
    col_log_scale = np.maximum(
        np.divide(
            np.where(
                valid,
                np.abs(log_distances - col_log_mean[None, :]),
                0.0,
            ).sum(axis=0),
            col_counts,
            out=np.ones(n, dtype=float),
            where=col_counts > 0.0,
        ),
        0.15,
    )

    row_z = (row_log_mean[:, None] - log_distances) / row_log_scale[:, None]
    col_z = (col_log_mean[None, :] - log_distances) / col_log_scale[None, :]

    row_gate = 1.0 / (1.0 + np.exp(-np.clip(row_z, -30.0, 30.0)))
    col_gate = 1.0 / (1.0 + np.exp(-np.clip(col_z, -30.0, 30.0)))
    endpoint_quality = np.sqrt(row_gate * col_gate)

    outgoing_after_edge = np.where(
        valid,
        np.broadcast_to(safe_distances[None, :, :], (n, n, n)),
        0.0,
    )

    continuation_valid = valid[None, :, :] & (
        np.arange(n)[None, :, None] != np.arange(n)[None, None, :]
    )
    continuation_sum = continuation_valid.astype(float)
    continuation_cost = np.divide(
        outgoing_after_edge.sum(axis=2),
        continuation_sum.sum(axis=2),
        out=np.broadcast_to(row_mean[None, :], (n, n)).copy(),
        where=continuation_sum.sum(axis=2) > 0.0,
    )

    destination_scale = np.maximum(col_mean, base_scale * eps)
    continuation_ratio = continuation_cost / destination_scale[None, :]
    continuation_quality = np.exp(
        -0.35 * np.clip(continuation_ratio, 0.0, 60.0)
    )

    source_scale = np.maximum(row_mean, base_scale * eps)
    edge_ratio = distances / np.sqrt(
        source_scale[:, None] * destination_scale[None, :]
    )
    smooth_distance = np.exp(
        -np.square(np.clip(edge_ratio, 0.0, 12.0))
    )

    dispersion_balance = 1.0 / (
        1.0
        + 0.20
        * (
            row_scale[:, None] / np.maximum(row_mean[:, None], eps)
            + col_scale[None, :] / np.maximum(col_mean[None, :], eps)
        )
    )

    scores = (
        smooth_distance
        * (0.35 + 0.65 * endpoint_quality)
        * (0.70 + 0.30 * continuation_quality)
        * dispersion_balance
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    row_totals = scores.sum(axis=1, keepdims=True)

    return np.divide(
        scores,
        row_totals,
        out=np.zeros_like(scores),
        where=row_totals > 0.0,
    ).astype(float, copy=False)
