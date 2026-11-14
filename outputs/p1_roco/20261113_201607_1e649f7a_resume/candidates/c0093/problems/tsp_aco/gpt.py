import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return spectral-global edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(raw)
    offdiag = ~np.eye(n, dtype=bool)
    valid = finite & offdiag

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(finite, np.maximum(raw, 0.0), 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)
    valid = np.isfinite(cleaned) & offdiag

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    distances = np.where(valid, cleaned, np.inf)
    finite_values = cleaned[valid]
    upper = float(np.max(finite_values)) if finite_values.size else scale

    temperature = max(
        scale * (0.55 + 0.20 * np.log1p(float(n))),
        np.finfo(float).eps,
    )

    conductance = np.zeros((n, n), dtype=float)
    conductance[valid] = np.exp(
        -np.minimum(distances[valid], upper + 12.0 * scale)
        / temperature
    )
    conductance = 0.5 * (conductance + conductance.T)
    conductance[~valid] = 0.0
    np.fill_diagonal(conductance, 0.0)

    degree = conductance.sum(axis=1)
    laplacian = np.diag(degree) - conductance

    regularizer = np.finfo(float).eps * max(float(degree.max()), 1.0)
    centered = laplacian + regularizer * np.eye(n)
    centered_pinv = np.linalg.pinv(centered, hermitian=True)

    effective_resistance = (
        np.diag(centered_pinv)[:, None]
        + np.diag(centered_pinv)[None, :]
        - 2.0 * centered_pinv
    )
    effective_resistance = np.maximum(effective_resistance, 0.0)
    effective_resistance[~valid] = 0.0
    np.fill_diagonal(effective_resistance, 0.0)

    resistance_values = effective_resistance[valid]
    resistance_scale = float(np.median(resistance_values))
    if not np.isfinite(resistance_scale) or resistance_scale <= 0.0:
        resistance_scale = 1.0

    global_bridge = 1.0 - np.exp(
        -effective_resistance / resistance_scale
    )

    common_support = conductance @ conductance
    support_scale = float(np.max(common_support[valid]))
    if not np.isfinite(support_scale) or support_scale <= 0.0:
        support_scale = 1.0
    common_support = common_support / support_scale

    row_mass = np.maximum(degree, np.finfo(float).eps)
    endpoint_balance = np.sqrt(
        row_mass[:, None] * row_mass[None, :]
    )
    endpoint_balance /= max(float(endpoint_balance.max()), np.finfo(float).eps)

    local = conductance / max(
        float(conductance.max()),
        np.finfo(float).eps,
    )

    score = (
        0.45 * local
        + 0.35 * global_bridge
        + 0.12 * (1.0 - common_support)
        + 0.08 * endpoint_balance
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
