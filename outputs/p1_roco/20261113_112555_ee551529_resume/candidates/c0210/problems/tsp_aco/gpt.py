import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-structural TSP edge priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.median(finite_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    # Combine directed observations without requiring both directions.
    affinity_distance = np.full((n, n), np.inf, dtype=float)
    transpose_valid = valid.T
    both = valid & transpose_valid
    forward_only = valid & ~transpose_valid
    backward_only = transpose_valid & ~valid

    affinity_distance[both] = 0.5 * (d[both] + d.T[both])
    affinity_distance[forward_only] = d[forward_only]
    affinity_distance[backward_only] = d.T[backward_only]
    np.fill_diagonal(affinity_distance, np.inf)

    usable = np.isfinite(affinity_distance)
    if not np.any(usable):
        return result

    # Bounded global affinity. This avoids local-scale and rank-based terms.
    normalized = np.divide(
        affinity_distance,
        max(scale, eps),
        out=np.full((n, n), np.inf, dtype=float),
        where=usable,
    )
    affinity = np.zeros((n, n), dtype=float)
    affinity[usable] = 1.0 / (1.0 + normalized[usable])

    # Row-stochastic transition structure.
    row_sums = affinity.sum(axis=1)
    transition = np.divide(
        affinity,
        row_sums[:, None],
        out=np.zeros_like(affinity),
        where=row_sums[:, None] > 0.0,
    )

    # Bidirectional probability flow captures mutual compatibility.
    reverse_flow = transition.T
    mutual_flow = np.sqrt(np.maximum(transition * reverse_flow, 0.0))

    # Two-step graph closure: edges supported by shared intermediate nodes
    # receive additional evidence.
    closure = transition @ transition
    closure = np.sqrt(np.maximum(closure * closure.T, 0.0))

    closure_max = float(np.max(closure))
    if np.isfinite(closure_max) and closure_max > eps:
        closure = closure / closure_max

    direct = np.sqrt(np.maximum(affinity * affinity.T, 0.0))
    direct_max = float(np.max(direct))
    if np.isfinite(direct_max) and direct_max > eps:
        direct = direct / direct_max

    score = (
        0.55 * mutual_flow
        + 0.30 * direct
        + 0.15 * direct * closure
    )

    score[~usable] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
