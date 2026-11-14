import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps

    # TSP edges are undirected; remove directional noise from the input.
    symmetric_d = 0.5 * (d + d.T)
    symmetric_valid = valid & valid.T
    np.fill_diagonal(symmetric_valid, False)

    safe_d = np.where(symmetric_valid, symmetric_d, np.inf)
    safe_d = np.where(
        np.isfinite(safe_d) & (safe_d > 0.0),
        safe_d,
        fallback,
    )
    np.fill_diagonal(safe_d, np.inf)

    k = min(3, n - 1)
    nearest = np.partition(safe_d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)

    # Reciprocal endpoint rankings provide a compact two-sided preference.
    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1).astype(float) + 1.0
    reciprocal_rank = 1.0 / np.sqrt(ranks * ranks.T)

    # Locally normalized short-edge preference.
    local_shortness = np.exp(
        -0.5 * safe_d * (1.0 / local_scale[:, None] + 1.0 / local_scale[None, :])
    )

    # Mild global term prevents isolated local scales from dominating.
    global_scale = float(np.median(nearest))
    global_shortness = np.exp(-safe_d / max(global_scale, eps))

    # Preserve structurally useful edges incident to sparse nodes.
    sparsity = local_scale / max(float(np.mean(local_scale)), eps)
    bridge_factor = np.sqrt(sparsity[:, None] * sparsity[None, :])
    bridge_factor = 0.85 + 0.15 * np.clip(bridge_factor, 0.0, 2.0)

    score = (
        (0.55 * reciprocal_rank + 0.45 * local_shortness)
        * (0.85 + 0.15 * global_shortness)
        * bridge_factor
    )

    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return score.astype(float, copy=False)
