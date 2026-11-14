import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global-structure edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    d = 0.5 * (distances + distances.T)
    symmetric_valid = valid & valid.T
    d = np.where(symmetric_valid, d, np.inf)
    np.fill_diagonal(d, np.inf)

    usable = np.isfinite(d)
    if not np.any(usable):
        return result

    positive = d[usable & (d > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    safe_distance = np.where(usable, np.maximum(d, 0.0), np.inf)

    affinity = np.zeros((n, n), dtype=float)
    affinity[usable] = (
        1.0
        / np.square(
            1.0 + safe_distance[usable] / scale
        )
    )
    np.fill_diagonal(affinity, 0.0)

    degree = affinity.sum(axis=1)
    safe_degree = np.maximum(degree, np.finfo(float).eps)

    normalized_affinity = affinity / np.sqrt(
        safe_degree[:, None] * safe_degree[None, :]
    )

    two_step = normalized_affinity @ normalized_affinity
    np.fill_diagonal(two_step, 0.0)

    edge_strength = affinity / np.sqrt(
        safe_degree[:, None] * safe_degree[None, :]
    )

    valid_two_step = two_step[usable]
    if valid_two_step.size:
        alternative_scale = float(np.median(valid_two_step))
    else:
        alternative_scale = 1.0

    if not np.isfinite(alternative_scale) or alternative_scale <= 0.0:
        alternative_scale = 1.0

    alternative_ratio = two_step / alternative_scale
    substitution_penalty = 1.0 / (
        1.0 + 0.85 * np.sqrt(np.maximum(alternative_ratio, 0.0))
    )

    endpoint_balance = 1.0 / np.sqrt(
        1.0 + 0.35 * safe_degree[:, None]
        + 0.35 * safe_degree[None, :]
    )

    direct_quality = np.sqrt(np.maximum(edge_strength, 0.0))
    exploration = 0.08 * affinity

    score = (
        (direct_quality + exploration)
        * substitution_penalty
        * endpoint_balance
    )

    score *= symmetric_valid
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
