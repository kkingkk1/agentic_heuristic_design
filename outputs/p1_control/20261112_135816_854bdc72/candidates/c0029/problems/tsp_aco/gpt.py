import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return prior indicators for promising TSP edges.

    The score combines:
    - robust inverse distance,
    - endpoint-local rank quality,
    - regret avoidance at both endpoints,
    - two-step continuation quality.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(d)
    if not np.all(finite):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    score = np.zeros_like(d, dtype=float)
    if n == 1:
        return score

    work = d.copy()
    np.fill_diagonal(work, np.nan)

    positive = work[np.isfinite(work) & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Robust inverse-cost affinity.
    affinity = 1.0 / (1.0 + work / scale)
    affinity[~np.isfinite(affinity)] = 0.0

    # Row-wise normalized neighborhood strength.
    row_max = np.max(affinity, axis=1, keepdims=True)
    row_max = np.maximum(row_max, np.finfo(float).eps)
    local_strength = affinity / row_max

    # Regret: selecting an edge is preferable when it does not consume
    # an endpoint's uniquely strong alternative.
    sorted_affinity = np.sort(affinity, axis=1)
    best = sorted_affinity[:, -1]
    second = sorted_affinity[:, -2] if n > 2 else best
    regret = np.maximum(best - second, 0.0)
    regret_scale = np.maximum(best, np.finfo(float).eps)
    endpoint_flexibility = 1.0 - regret / regret_scale

    # Two-step lookahead: how well each endpoint can continue after this edge.
    continuation = affinity @ affinity
    continuation_max = np.max(continuation, axis=1, keepdims=True)
    continuation_max = np.maximum(continuation_max, np.finfo(float).eps)
    continuation = continuation / continuation_max

    endpoint_flexibility_matrix = (
        endpoint_flexibility[:, None] + endpoint_flexibility[None, :]
    ) / 2.0

    local_matrix = (local_strength + local_strength.T) / 2.0
    continuation_matrix = (continuation + continuation.T) / 2.0

    score = (
        affinity ** 1.5
        * (0.55 + 0.45 * local_matrix)
        * (0.60 + 0.40 * endpoint_flexibility_matrix)
        * (0.70 + 0.30 * continuation_matrix)
    )

    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
