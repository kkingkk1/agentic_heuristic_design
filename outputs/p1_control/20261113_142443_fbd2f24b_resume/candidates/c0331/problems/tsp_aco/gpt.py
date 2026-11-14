import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-promise indicators using soft neighborhood propagation."""
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

    symmetric_d = 0.5 * (d + d.T)
    symmetric_d = np.where(valid, symmetric_d, np.inf)

    positive = symmetric_d[valid & (symmetric_d > 0.0)]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
        upper_scale = float(np.percentile(positive, 75.0))
    else:
        scale = 1.0
        upper_scale = 1.0

    scale = max(scale, eps)
    upper_scale = max(upper_scale, scale, eps)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        direct = np.exp(-symmetric_d / upper_scale)

    direct = np.where(valid, direct, 0.0)
    np.fill_diagonal(direct, 0.0)

    row_mass = np.sum(direct, axis=1)
    row_mass = np.maximum(row_mass, eps)

    transition = direct / row_mass[:, None]
    transition = np.where(valid, transition, 0.0)

    reverse_transition = transition.T

    endpoint_support = np.sqrt(
        np.maximum(transition, 0.0)
        * np.maximum(reverse_transition, 0.0)
    )

    common_support = transition @ reverse_transition
    common_support = np.maximum(common_support, 0.0)

    two_step_forward = transition @ transition
    two_step_backward = reverse_transition @ reverse_transition
    bridge_support = np.sqrt(
        np.maximum(two_step_forward, 0.0)
        * np.maximum(two_step_backward, 0.0)
    )

    max_common = float(np.max(common_support))
    if max_common > 0.0 and np.isfinite(max_common):
        common_support = common_support / max_common

    max_bridge = float(np.max(bridge_support))
    if max_bridge > 0.0 and np.isfinite(max_bridge):
        bridge_support = bridge_support / max_bridge

    endpoint_strength = np.sqrt(
        np.maximum(row_mass, eps)[:, None]
        * np.maximum(row_mass, eps)[None, :]
    )
    endpoint_strength /= max(float(np.max(endpoint_strength)), eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        direct_quality = np.sqrt(
            np.maximum(direct, 0.0)
            * np.maximum(direct.T, 0.0)
        )

        score = (
            direct_quality ** 0.72
            * (0.55 + 0.45 * endpoint_support)
            * (0.62 + 0.38 * common_support)
            * (0.78 + 0.22 * bridge_support)
            * (0.86 + 0.14 * endpoint_strength)
        )

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
