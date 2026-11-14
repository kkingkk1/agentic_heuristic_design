import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed stochastic edge priorities from triangle-closure gains."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    positive = d[valid & (d > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    safe_d = np.where(valid, d, np.inf)

    finite_row_values = np.where(valid, d, np.nan)
    row_scale = np.nanmedian(finite_row_values, axis=1)
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0.0),
        row_scale,
        global_scale,
    )

    finite_col_values = np.where(valid, d, np.nan)
    col_scale = np.nanmedian(finite_col_values, axis=0)
    col_scale = np.where(
        np.isfinite(col_scale) & (col_scale > 0.0),
        col_scale,
        global_scale,
    )

    # Normalize each directed edge by the scale of its source and destination.
    edge_scale = np.sqrt(row_scale[:, None] * col_scale[None, :])
    direct_ratio = np.divide(
        d,
        np.maximum(edge_scale, np.finfo(float).eps),
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    # Compute direct-edge advantage over all available two-hop alternatives.
    closure_values = np.zeros((n, n), dtype=float)
    closure_counts = np.zeros((n, n), dtype=float)

    for k in range(n):
        through_k = safe_d[:, k, None] + safe_d[None, k, :]
        usable = np.isfinite(through_k) & valid
        gain = through_k - d
        gain = np.divide(
            gain,
            np.maximum(edge_scale, np.finfo(float).eps),
            out=np.zeros_like(d, dtype=float),
            where=usable,
        )
        gain = np.clip(gain, -20.0, 20.0)
        closure_values += np.where(usable, gain, 0.0)
        closure_counts += usable

    mean_gain = np.divide(
        closure_values,
        closure_counts,
        out=np.zeros_like(d, dtype=float),
        where=closure_counts > 0,
    )

    # Favor edges with both broad and strong two-hop superiority.
    closure_strength = 1.0 / (1.0 + np.exp(-mean_gain))

    # A weak direct-distance factor prevents very long edges from dominating
    # when triangle evidence is sparse.
    direct_factor = np.exp(-np.clip(0.35 * direct_ratio, 0.0, 30.0))

    # Reward edges supported by at least one especially poor detour.
    best_detour_gain = np.divide(
        closure_values,
        np.maximum(closure_counts, 1.0),
        out=np.zeros_like(d, dtype=float),
        where=closure_counts > 0,
    )
    detour_bonus = 1.0 + 0.20 * np.tanh(np.maximum(best_detour_gain, 0.0))

    logits = (
        1.8 * np.log(np.maximum(closure_strength, 1e-12))
        + np.log(np.maximum(direct_factor, 1e-12))
        + np.log(np.maximum(detour_bonus, 1e-12))
    )
    logits[~valid] = -np.inf
    np.fill_diagonal(logits, -np.inf)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        row_logits = logits[i, row_valid]
        row_logits -= np.max(row_logits)
        weights = np.exp(np.clip(row_logits, -60.0, 0.0))
        total = np.sum(weights)

        if np.isfinite(total) and total > 0.0:
            result[i, row_valid] = weights / total

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
