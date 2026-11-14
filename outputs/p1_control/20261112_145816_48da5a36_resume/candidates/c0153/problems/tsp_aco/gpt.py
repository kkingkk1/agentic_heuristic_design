import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric diffusion-based edge-promise matrix for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    d = np.where(finite, d, np.inf)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    scale = max(float(np.median(positive_values)), eps)

    normalized = np.zeros_like(d, dtype=float)
    normalized[valid] = d[valid] / scale

    direct = np.zeros_like(d, dtype=float)
    direct[valid] = np.exp(-np.clip(normalized[valid], 0.0, 745.0))

    row_mass = np.sum(direct, axis=1, keepdims=True)
    transition = np.divide(
        direct,
        row_mass,
        out=np.zeros_like(direct),
        where=row_mass > 0.0,
    )

    mutual_flow = np.sqrt(
        np.maximum(transition * transition.T, 0.0)
    )

    two_step = transition @ transition
    common_support = np.sqrt(
        np.maximum(two_step * two_step.T, 0.0)
    )

    direct_peak = np.max(direct[valid])
    direct_component = np.zeros_like(direct)
    if direct_peak > 0.0:
        direct_component[valid] = direct[valid] / direct_peak

    flow_peak = np.max(mutual_flow[valid])
    flow_component = np.zeros_like(mutual_flow)
    if flow_peak > 0.0:
        flow_component[valid] = mutual_flow[valid] / flow_peak

    support_peak = np.max(common_support[valid])
    support_component = np.zeros_like(common_support)
    if support_peak > 0.0:
        support_component[valid] = common_support[valid] / support_peak

    score = (
        0.45 * direct_component
        + 0.35 * flow_component
        + 0.20 * support_component
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
