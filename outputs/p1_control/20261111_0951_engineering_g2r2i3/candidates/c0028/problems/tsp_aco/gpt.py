import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    safe = np.where(valid, d, np.inf)

    finite_values = safe[valid]
    if finite_values.size:
        low = float(np.percentile(finite_values, 20.0))
        high = float(np.percentile(finite_values, 80.0))
        temperature = max(high - low, 0.25 * global_scale, eps)
    else:
        temperature = global_scale

    kernel = np.zeros((n, n), dtype=float)
    kernel[valid] = np.exp(
        -np.minimum(d[valid] / temperature, 60.0)
    )

    np.fill_diagonal(kernel, 0.0)

    # Entropy-regularized balancing prevents dense rows from dominating.
    balanced = kernel.copy()
    for _ in range(40):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=col_sum > 0.0,
        )

    balanced = 0.5 * (balanced + balanced.T)

    # Measure whether the direct edge is competitive with two-hop alternatives.
    finite_for_min = np.where(valid, d, np.inf)
    two_hop = finite_for_min[:, :, None] + finite_for_min[None, :, :]
    two_hop = np.min(two_hop, axis=1)

    direct = np.where(valid, d, np.inf)
    detour_margin = two_hop - direct

    closure = np.zeros((n, n), dtype=float)
    closure[valid] = 1.0 / (
        1.0 + np.exp(
            -np.clip(detour_margin[valid] / temperature, -30.0, 30.0)
        )
    )

    # Downweight edges for which a comparable two-hop route is clearly better.
    score = balanced * (0.35 + 0.65 * closure)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
