import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric prior based on an entropy-regularized 2-regular relaxation."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_edges = np.isfinite(d)
    positive = d[finite_edges & (d > 0)]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    # Temperature is chosen from the global distance dispersion.
    spread = float(np.median(np.abs(positive - scale))) if positive.size else 0.0
    temperature = max(scale + spread, np.finfo(float).eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[finite_edges] = np.exp(
        -np.minimum(d[finite_edges] / temperature, 60.0)
    )
    np.fill_diagonal(affinity, 0.0)

    if not np.any(affinity > 0):
        return np.zeros((n, n), dtype=float)

    # Symmetric iterative scaling toward a soft degree-two constraint.
    scaling = np.ones(n, dtype=float)
    for _ in range(80):
        weighted_degree = scaling * (affinity @ scaling)
        valid = weighted_degree > 0
        update = np.ones(n, dtype=float)
        update[valid] = np.sqrt(2.0 / weighted_degree[valid])
        update = np.clip(update, 1e-8, 1e8)

        if np.max(np.abs(update - 1.0)) < 1e-8:
            break

        scaling *= update
        scaling = np.clip(scaling, 1e-8, 1e8)

    relaxed = scaling[:, None] * affinity * scaling[None, :]
    relaxed = 0.5 * (relaxed + relaxed.T)
    np.fill_diagonal(relaxed, 0.0)

    # Common-neighbor mass measures whether an edge participates in
    # multiple short two-step closures under the relaxed 2-regular graph.
    closure = relaxed @ relaxed
    closure = 0.5 * (closure + closure.T)

    row_mass = np.sum(relaxed, axis=1)
    normalization = np.sqrt(
        np.maximum(row_mass[:, None] * row_mass[None, :], 1e-30)
    )
    closure_ratio = closure / normalization
    closure_ratio = np.clip(closure_ratio, 0.0, 1.0)

    score = relaxed * (0.65 + 0.35 * closure_ratio)
    score[~finite_edges] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
