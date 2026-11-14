import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-priority indicators for stochastic TSP solution sampling.
    Larger values indicate more promising directed edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d)
    valid = finite & ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    safe_d = np.where(valid, np.maximum(d, 0.0), np.inf)

    positive = safe_d[valid & (safe_d > 0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        finite_values = safe_d[valid]
        scale = float(np.median(finite_values)) if finite_values.size else 1.0

    scale = max(scale, 1e-12)

    # Bounded proximity is more stable than direct reciprocal distance.
    proximity = np.zeros_like(safe_d, dtype=float)
    proximity[valid] = scale / (safe_d[valid] + scale)

    # Robust normalization of the direct quality signal.
    proximity_max = np.max(proximity)
    if proximity_max > 0:
        proximity /= proximity_max

    # Find the nearest and second-nearest valid edge for every node.
    ranked = np.argpartition(safe_d, kth=min(1, n - 1), axis=1)
    nearest = ranked[:, 0]
    second_nearest = ranked[:, min(1, n - 1)]

    row_index = np.arange(n)
    nearest_distance = safe_d[row_index, nearest]

    # Savings signal: retaining edge (i, j) avoids two endpoint alternatives.
    endpoint_baseline = (
        second_nearest[:, None] + second_nearest[None, :]
    )
    savings = np.zeros_like(safe_d, dtype=float)
    finite_savings = (
        valid
        & np.isfinite(endpoint_baseline)
        & np.isfinite(safe_d)
    )
    savings[finite_savings] = np.maximum(
        endpoint_baseline[finite_savings] - safe_d[finite_savings],
        0.0,
    )

    savings_max = np.max(savings)
    if savings_max > 0:
        savings /= savings_max

    # Reciprocal compatibility rewards edges that are locally attractive
    # from both endpoints, rather than only from their source node.
    reciprocal = 0.5 * (proximity + proximity.T)

    mutual = np.zeros_like(safe_d, dtype=float)
    nearest_target = nearest
    mutual_mask = (
        valid
        & (nearest_target[row_index] != row_index)
        & (nearest_target[nearest_target] == row_index)
    )
    mutual[mutual_mask] = 1.0

    # Adaptive local candidate retention. The union of row and column
    # neighborhoods preserves directed exploration in asymmetric instances.
    neighborhood_size = max(
        2,
        min(n - 1, int(np.ceil(np.sqrt(n))) + 1),
    )

    row_cutoff = np.partition(
        safe_d,
        kth=min(neighborhood_size - 1, n - 1),
        axis=1,
    )[:, min(neighborhood_size - 1, n - 1)]

    local_candidate = valid & (
        (safe_d <= row_cutoff[:, None])
        | (safe_d.T <= row_cutoff[:, None])
    )

    # Always retain mutual nearest-neighbor edges.
    candidate = local_candidate | mutual_mask

    # Preserve a small global elite set so local pruning cannot remove
    # structurally useful long-range edges.
    base_score = (
        0.36 * savings
        + 0.30 * reciprocal
        + 0.24 * proximity
        + 0.10 * mutual
    )
    base_score[~valid] = 0.0

    valid_scores = base_score[valid]
    if valid_scores.size:
        elite_count = max(1, min(valid_scores.size, 2 * n))
        flat_valid = np.flatnonzero(valid)
        elite_order = np.argsort(base_score[valid], kind="stable")[-elite_count:]
        elite_mask = np.zeros(n * n, dtype=bool)
        elite_mask[flat_valid[elite_order]] = True
        candidate |= elite_mask.reshape(n, n)

    # Keep exploration mass on pruned edges while strongly favoring candidates.
    score = np.where(candidate, base_score, 0.035 * base_score)

    # Add a small quality floor among valid edges to avoid deterministic
    # collapse when many edges have tied or zero component scores.
    score = np.where(valid, score + 0.01 * proximity, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
