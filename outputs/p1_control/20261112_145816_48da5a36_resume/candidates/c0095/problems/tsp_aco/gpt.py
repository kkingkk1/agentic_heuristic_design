import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)

    safe_d = np.where(valid, d, np.inf)

    # Rational conductance avoids the rank-based and exponential local-scale
    # constructions used by the parent heuristics.
    scaled = safe_d / global_scale
    conductance = np.zeros((n, n), dtype=float)
    conductance[valid] = 1.0 / (1.0 + scaled[valid] ** 2)

    # Normalize conductance by endpoint weighted degrees.
    degree = np.sum(conductance, axis=1)
    degree = np.maximum(degree, eps)
    normalized_conductance = conductance / np.sqrt(
        degree[:, None] * degree[None, :]
    )

    # Effective resistance captures whether an edge structurally connects
    # otherwise weakly connected parts of the graph.
    laplacian = np.diag(degree) - conductance
    try:
        laplacian_pinv = np.linalg.pinv(laplacian, rcond=1e-10)
        resistance = (
            np.diag(laplacian_pinv)[:, None]
            + np.diag(laplacian_pinv)[None, :]
            - 2.0 * laplacian_pinv
        )
        resistance = np.maximum(resistance, 0.0)
        resistance[~np.isfinite(resistance)] = 0.0
    except np.linalg.LinAlgError:
        resistance = np.zeros((n, n), dtype=float)

    valid_resistance = resistance[valid]
    if valid_resistance.size:
        resistance_scale = max(
            float(np.percentile(valid_resistance, 90.0)),
            eps,
        )
        resistance_signal = np.clip(
            resistance / resistance_scale,
            0.0,
            1.0,
        )
    else:
        resistance_signal = np.zeros((n, n), dtype=float)

    # Compute the cheapest two-hop replacement for each edge.
    two_hop = np.min(
        safe_d[:, :, None] + safe_d[None, :, :],
        axis=1,
    )
    np.fill_diagonal(two_hop, np.inf)

    shortcut_ratio = np.zeros((n, n), dtype=float)
    finite_two_hop = valid & np.isfinite(two_hop)
    shortcut_ratio[finite_two_hop] = (
        two_hop[finite_two_hop]
        / np.maximum(d[finite_two_hop], eps)
    )

    # Compress extreme ratios while preserving the ordering of useful
    # direct shortcuts.
    shortcut_signal = np.zeros((n, n), dtype=float)
    shortcut_signal[finite_two_hop] = np.log1p(
        np.maximum(shortcut_ratio[finite_two_hop] - 1.0, 0.0)
    )
    shortcut_values = shortcut_signal[finite_two_hop]
    if shortcut_values.size:
        shortcut_scale = max(float(np.percentile(shortcut_values, 90.0)), eps)
        shortcut_signal = np.clip(shortcut_signal / shortcut_scale, 0.0, 1.0)

    score = (
        normalized_conductance
        * (0.70 + 0.30 * resistance_signal)
        * (0.75 + 0.25 * shortcut_signal)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
