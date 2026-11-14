import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic directed edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    observed = d[valid]
    positive = observed[observed > 0.0]

    center = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(center) or center <= 0.0:
        center = 1.0

    q10, q50, q90 = np.quantile(observed, [0.10, 0.50, 0.90])
    spread = max(float(q90 - q10), center * 1e-6)
    invalid_value = max(float(q90), float(q50)) + 20.0 * spread

    clean = np.where(valid, d, invalid_value).astype(float, copy=True)
    np.fill_diagonal(clean, invalid_value)

    eps = np.finfo(float).eps
    safe_spread = max(spread, eps)

    row_sorted = np.sort(clean, axis=1)
    col_sorted = np.sort(clean, axis=0)

    row_median = row_sorted[:, n // 2]
    col_median = col_sorted[n // 2, :]

    row_scale = np.maximum(
        np.median(np.abs(clean - row_median[:, None]), axis=1),
        safe_spread * 0.05,
    )
    col_scale = np.maximum(
        np.median(np.abs(clean - col_median[None, :]), axis=0),
        safe_spread * 0.05,
    )

    row_z = (clean - row_median[:, None]) / row_scale[:, None]
    col_z = (clean - col_median[None, :]) / col_scale[None, :]

    row_affinity = np.exp(-np.clip(row_z, -12.0, 12.0))
    col_affinity = np.exp(-np.clip(col_z, -12.0, 12.0))

    row_affinity[~valid] = 0.0
    col_affinity[~valid] = 0.0

    row_order = np.argsort(clean, axis=1, kind="stable")
    col_order = np.argsort(clean, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    rank_scale = max(1.0, np.sqrt(float(n)))
    row_rank_score = np.exp(-row_rank / rank_scale)
    col_rank_score = np.exp(-col_rank / rank_scale)

    row_rank_score[~valid] = 0.0
    col_rank_score[~valid] = 0.0

    outgoing_options = np.sum(valid, axis=1).astype(float)
    incoming_options = np.sum(valid, axis=0).astype(float)

    outgoing_flexibility = np.maximum(
        outgoing_options / max(float(n - 1), 1.0),
        1.0 / max(float(n - 1), 1.0),
    )
    incoming_flexibility = np.maximum(
        incoming_options / max(float(n - 1), 1.0),
        1.0 / max(float(n - 1), 1.0),
    )

    destination_access = outgoing_flexibility[None, :]
    source_pressure = incoming_flexibility[:, None]

    local_quality = np.sqrt(row_affinity * col_affinity)
    rank_agreement = np.sqrt(row_rank_score * col_rank_score)

    scarcity = 1.0 / (
        1.0
        + np.maximum(row_rank, 0.0)
        + np.maximum(col_rank, 0.0)
    )
    scarcity *= 1.0 + 0.5 * (
        (1.0 - destination_access)
        + (1.0 - source_pressure)
    )

    directed_score = (
        0.46 * local_quality
        + 0.24 * rank_agreement
        + 0.20 * scarcity
        + 0.10 * np.sqrt(
            np.maximum(destination_access * source_pressure, 0.0)
        )
    )

    directed_score[~valid] = 0.0

    weights = directed_score.copy()
    for _ in range(10):
        row_sum = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(row_sum, eps)

        col_sum = weights.sum(axis=0, keepdims=True)
        weights /= np.maximum(col_sum, eps)

        weights = 0.72 * weights + 0.28 * directed_score

    result = (
        0.62 * weights
        + 0.23 * local_quality
        + 0.15 * rank_agreement
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
