import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global-balance and path-closure edge priors for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_offdiag = np.isfinite(distances) & ~diagonal

    if not np.any(finite_offdiag):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[finite_offdiag]
    finite_values = np.maximum(finite_values, 0.0)

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.mean(finite_values)) if finite_values.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    spread = float(np.median(np.abs(finite_values - np.median(finite_values))))
    if not np.isfinite(spread) or spread <= 0.0:
        spread = 0.25 * scale

    temperature = max(0.15 * scale, spread, np.finfo(float).eps)

    finite_max = float(np.max(finite_values))
    replacement = finite_max + 20.0 * max(scale, 1.0)

    clean = np.where(
        finite_offdiag,
        np.maximum(distances, 0.0),
        replacement,
    ).astype(float, copy=False)

    np.fill_diagonal(clean, replacement)

    logits = -clean / temperature
    logits = np.clip(logits, -60.0, 0.0)
    affinity = np.exp(logits)
    affinity[~finite_offdiag] = 0.0

    # Sinkhorn balancing produces a globally degree-aware edge flow.
    flow = affinity.copy()
    tiny = np.finfo(float).tiny

    for _ in range(80):
        row_sum = flow.sum(axis=1, keepdims=True)
        flow /= np.maximum(row_sum, tiny)

        col_sum = flow.sum(axis=0, keepdims=True)
        flow /= np.maximum(col_sum, tiny)

        flow[~finite_offdiag] = 0.0

    # Measure whether a direct edge is difficult to replace by a two-step path.
    via = clean[:, :, None] + clean[None, :, :]
    via[:, np.arange(n), :] = replacement
    via[:, :, np.arange(n)] = replacement

    best_two_step = np.min(via, axis=1)
    direct_advantage = best_two_step - clean

    closure_scale = max(scale, temperature, np.finfo(float).eps)
    closure_pressure = 1.0 / (
        1.0 + np.exp(-np.clip(direct_advantage / closure_scale, -30.0, 30.0))
    )
    closure_pressure[~finite_offdiag] = 0.0

    # Edges with globally concentrated flow receive a mild scarcity bonus.
    row_mass = flow.sum(axis=1, keepdims=True)
    col_mass = flow.sum(axis=0, keepdims=True)
    scarcity = flow / np.sqrt(np.maximum(row_mass * col_mass, tiny))
    scarcity[~finite_offdiag] = 0.0

    raw = (
        0.62 * flow
        + 0.23 * closure_pressure
        + 0.15 * scarcity
    )

    raw[~finite_offdiag] = 0.0
    np.fill_diagonal(raw, 0.0)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(raw))
    if np.isfinite(maximum) and maximum > 0.0:
        raw /= maximum

    return raw.astype(float, copy=False)
