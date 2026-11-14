import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a graph-diffusion-based edge-priority matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric_distance = np.zeros((n, n), dtype=float)
    symmetric_distance[valid] = 0.5 * (
        d[valid] + d.T[valid]
    )

    positive_values = symmetric_distance[
        valid & (symmetric_distance > 0.0)
    ]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    safe_distance = np.maximum(symmetric_distance, 0.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(safe_distance[valid] / global_scale, 50.0)
    )

    degree = affinity.sum(axis=1)
    degree_safe = np.maximum(degree, eps)

    laplacian = np.diag(degree) - affinity

    try:
        laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
        effective_resistance = (
            np.diag(laplacian_pinv)[:, None]
            + np.diag(laplacian_pinv)[None, :]
            - 2.0 * laplacian_pinv
        )
        effective_resistance = np.maximum(effective_resistance, 0.0)
    except np.linalg.LinAlgError:
        effective_resistance = np.zeros((n, n), dtype=float)

    resistance_values = effective_resistance[valid]
    positive_resistance = resistance_values[resistance_values > eps]
    resistance_scale = (
        float(np.median(positive_resistance))
        if positive_resistance.size
        else 1.0
    )
    resistance_scale = max(resistance_scale, eps)

    bridge_contribution = effective_resistance / resistance_scale
    bridge_contribution = np.minimum(bridge_contribution, 20.0)

    transition = affinity / degree_safe[:, None]
    two_step_support = transition @ transition
    two_step_support = 0.5 * (
        two_step_support + two_step_support.T
    )
    two_step_support = np.maximum(two_step_support, 0.0)
    two_step_support = np.minimum(two_step_support, 1.0)

    degree_ratio = np.abs(
        np.log(degree_safe[:, None] / degree_safe[None, :])
    )
    degree_balance = np.exp(-0.12 * np.minimum(degree_ratio, 30.0))

    direct_component = np.power(
        np.maximum(affinity, eps),
        0.85,
    )
    bridge_component = 1.0 + 0.55 * (
        bridge_contribution / (1.0 + bridge_contribution)
    )
    closure_component = 0.72 + 0.58 * two_step_support

    score = (
        direct_component
        * bridge_component
        * closure_component
        * degree_balance
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
