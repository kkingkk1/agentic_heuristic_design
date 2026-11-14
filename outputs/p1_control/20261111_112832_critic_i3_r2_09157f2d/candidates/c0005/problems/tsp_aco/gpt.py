import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = d[finite_nonnegative]

    if valid.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = valid[valid > 0.0]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    max_valid = float(np.max(valid))
    fallback = max(max_valid, global_scale) * 4.0 + global_scale

    clean = np.where(finite_nonnegative, d, fallback)
    clean = np.maximum(clean, 0.0)

    symmetric = np.minimum(clean, clean.T)
    np.fill_diagonal(symmetric, fallback)

    bandwidth = np.empty(n, dtype=float)
    for i in range(n):
        row = symmetric[i, np.arange(n) != i]
        positive_row = row[row > 0.0]
        if positive_row.size:
            value = float(np.median(positive_row))
        else:
            value = global_scale
        bandwidth[i] = value if np.isfinite(value) and value > 0.0 else global_scale

    pair_scale = np.sqrt(np.outer(bandwidth, bandwidth))
    pair_scale = np.maximum(pair_scale, global_scale * 1e-12)

    affinity = np.exp(-np.square(symmetric / pair_scale))
    np.fill_diagonal(affinity, 0.0)

    row_mass = affinity.sum(axis=1)
    transition = np.divide(
        affinity,
        row_mass[:, None],
        out=np.zeros_like(affinity),
        where=row_mass[:, None] > 0.0,
    )

    diffusion = transition @ transition
    diffusion = 0.5 * (diffusion + diffusion.T)

    endpoint_mass = np.sqrt(np.outer(row_mass, row_mass))
    score = diffusion * endpoint_mass

    affinity_sym = 0.5 * (affinity + affinity.T)
    score *= np.sqrt(np.maximum(affinity_sym, 0.0))

    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
