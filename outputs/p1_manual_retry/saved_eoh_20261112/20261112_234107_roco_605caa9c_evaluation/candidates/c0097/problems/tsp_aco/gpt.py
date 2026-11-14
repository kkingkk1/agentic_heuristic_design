import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a graph-normalized structural prior for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(d) & off_diagonal
    nonnegative = d >= 0.0
    valid = finite & nonnegative

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    # Construct a symmetric cost matrix while retaining one-sided valid edges.
    reverse_valid = valid.T
    both_valid = valid & reverse_valid
    only_forward = valid & ~reverse_valid
    only_reverse = reverse_valid & ~valid

    symmetric_cost = np.full((n, n), np.inf, dtype=float)
    symmetric_cost[both_valid] = 0.5 * (
        d[both_valid] + d.T[both_valid]
    )
    symmetric_cost[only_forward] = d[only_forward]
    symmetric_cost[only_reverse] = d.T[only_reverse]
    np.fill_diagonal(symmetric_cost, np.inf)

    usable_costs = symmetric_cost[np.isfinite(symmetric_cost)]
    if usable_costs.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = usable_costs[usable_costs > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        center = float(np.median(positive))
        spread = float(np.percentile(positive, 75.0) -
                       np.percentile(positive, 25.0))
        temperature = center + 0.5 * spread
    else:
        temperature = max(float(np.median(usable_costs)), 1.0)

    temperature = max(temperature, eps)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        affinity = np.exp(
            -np.clip(symmetric_cost / temperature, 0.0, 60.0)
        )

    affinity[~np.isfinite(affinity)] = 0.0
    affinity[~np.isfinite(symmetric_cost)] = 0.0
    np.fill_diagonal(affinity, 0.0)

    # Symmetric degree normalization, analogous to normalized graph kernels.
    degree = np.sum(affinity, axis=1)
    degree = np.maximum(degree, eps)
    normalized = affinity / np.sqrt(
        degree[:, None] * degree[None, :]
    )
    normalized = np.nan_to_num(
        normalized, nan=0.0, posinf=0.0, neginf=0.0
    )

    # Reward edges whose endpoints share support in the affinity graph.
    common_support = normalized @ normalized
    common_support[~np.isfinite(common_support)] = 0.0

    support_values = common_support[off_diagonal & (normalized > 0.0)]
    if support_values.size:
        support_scale = max(float(np.median(support_values)), eps)
    else:
        support_scale = 1.0

    structural_bonus = 1.0 + 0.35 * np.tanh(
        common_support / support_scale
    )

    score = normalized * structural_bonus
    score = 0.5 * (score + score.T)
    score[~valid & ~valid.T] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(
        score, nan=0.0, posinf=0.0, neginf=0.0
    )

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
