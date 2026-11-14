import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-prior indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    safe = np.where(valid, d, np.nan)

    row_min = np.nanmin(safe, axis=1)
    row_scale = np.nanmedian(
        np.where(valid, np.abs(d - row_min[:, None]), np.nan),
        axis=1,
    )
    global_scale = np.nanmedian(np.abs(d[valid]))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        global_scale,
    )

    # Basic distance advantage relative to the source node's alternatives.
    relative_cost = (d - row_min[:, None]) / (row_scale[:, None] + 1e-12)
    distance_advantage = np.exp(-np.clip(relative_cost, -50.0, 50.0))

    # Destination-side scarcity: an edge is more useful when its destination
    # has comparatively few inexpensive incoming alternatives.
    incoming = np.where(valid, d, np.nan)
    col_min = np.nanmin(incoming, axis=0)
    col_scale = np.nanmedian(
        np.where(valid, np.abs(d - col_min[None, :]), np.nan),
        axis=0,
    )
    col_scale = np.where(
        np.isfinite(col_scale) & (col_scale > 0),
        col_scale,
        global_scale,
    )

    destination_scarcity = np.exp(
        -np.clip((d - col_min[None, :]) / (col_scale[None, :] + 1e-12), -50.0, 50.0)
    )

    # Opportunity pressure estimates how much worse the best alternative is
    # after excluding the current edge at each endpoint.
    row_sorted = np.sort(np.where(valid, d, np.inf), axis=1)
    row_best = row_sorted[:, 0]
    row_second = row_sorted[:, 1] if n > 1 else row_best

    col_sorted = np.sort(np.where(valid, d, np.inf), axis=0)
    col_best = col_sorted[0, :]
    col_second = col_sorted[1, :] if n > 1 else col_best

    source_irreplaceability = np.maximum(
        row_second[:, None] - d,
        0.0,
    ) / (row_scale[:, None] + 1e-12)

    destination_irreplaceability = np.maximum(
        col_second[None, :] - d,
        0.0,
    ) / (col_scale[None, :] + 1e-12)

    scarcity = 1.0 + np.tanh(
        0.5 * (source_irreplaceability + destination_irreplaceability)
    )

    # Penalize edges that are expensive relative to both endpoint baselines.
    detour_pressure = np.maximum(
        (d - row_best[:, None]) / (row_scale[:, None] + 1e-12),
        0.0,
    ) + np.maximum(
        (d - col_best[None, :]) / (col_scale[None, :] + 1e-12),
        0.0,
    )
    detour_penalty = np.exp(-0.35 * np.clip(detour_pressure, 0.0, 100.0))

    score = (
        np.sqrt(distance_advantage * destination_scarcity)
        * scarcity
        * detour_penalty
    )
    score = np.where(valid & np.isfinite(score), score, 0.0)

    # Normalize outgoing indicators while retaining relative strength.
    row_sum = score.sum(axis=1, keepdims=True)
    result = np.divide(
        score,
        row_sum,
        out=np.zeros_like(score),
        where=row_sum > 0,
    )

    return result
