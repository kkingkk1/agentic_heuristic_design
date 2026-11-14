import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0

    safe_d = np.where(valid, d, np.inf)
    safe_d = np.where(safe_d > 0.0, safe_d, fallback)
    np.fill_diagonal(safe_d, np.inf)

    # Node-specific distance scale makes the score robust to heterogeneous regions.
    sorted_distances = np.sort(safe_d, axis=1)
    k = min(3, max(1, n - 1))
    local_scale = np.mean(sorted_distances[:, :k], axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Basic short-edge affinity with local-scale normalization.
    short_edge = np.exp(-safe_d / local_scale[:, None])

    # Mutual-neighborhood affinity favors edges promising from both endpoints.
    reverse_scale = local_scale[None, :]
    mutual_affinity = np.exp(
        -safe_d * 0.5 * (1.0 / local_scale[:, None] + 1.0 / reverse_scale)
    )

    # Bridge bonus preserves useful connections from locally sparse nodes.
    sparsity = local_scale / (np.mean(local_scale) + np.finfo(float).eps)
    bridge_bonus = np.sqrt(sparsity[:, None] * sparsity[None, :])

    score = short_edge * (0.55 + 0.45 * mutual_affinity) * (
        0.75 + 0.25 * bridge_bonus
    )

    # Suppress self-loops and invalid edges.
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    # Row normalization keeps each node's outgoing indicators comparable.
    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return score.astype(float, copy=False)
