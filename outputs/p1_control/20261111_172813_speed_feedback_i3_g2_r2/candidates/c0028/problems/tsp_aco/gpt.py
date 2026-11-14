import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return entropically balanced edge-priority indicators for TSP sampling.
    Larger values indicate more promising undirected edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    finite_values = d[valid]
    spread = float(np.percentile(finite_values, 75) -
                   np.percentile(finite_values, 25))
    temperature = max(0.5 * scale + 0.5 * spread,
                      np.finfo(float).eps)

    kernel = np.zeros((n, n), dtype=float)
    scaled = np.clip(d[valid] / temperature, 0.0, 700.0)
    kernel[valid] = np.exp(-scaled)

    # Iterative proportional fitting balances outgoing and incoming mass.
    # Zero-support rows or columns remain zero and do not create NaNs.
    for _ in range(80):
        row_mass = kernel.sum(axis=1)
        row_scale = np.ones(n, dtype=float)
        active_rows = row_mass > 0.0
        row_scale[active_rows] = 1.0 / row_mass[active_rows]
        kernel *= row_scale[:, None]

        col_mass = kernel.sum(axis=0)
        col_scale = np.ones(n, dtype=float)
        active_cols = col_mass > 0.0
        col_scale[active_cols] = 1.0 / col_mass[active_cols]
        kernel *= col_scale[None, :]

    score = 0.5 * (kernel + kernel.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
