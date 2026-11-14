import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return contrastive local-competition priors for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d)
    off_diagonal = ~np.eye(n, dtype=bool)
    finite_values = d[valid & off_diagonal]

    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(finite_values) + np.ptp(finite_values) + 1.0)
    clean = np.where(valid, d, fallback)
    clean = np.maximum(clean, 0.0)
    np.fill_diagonal(clean, fallback)

    positive = finite_values[finite_values > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Basic distance affinity.
    affinity = np.exp(-clean / scale)

    # Row and column contrasts measure competitiveness against alternatives.
    row_best = np.min(clean, axis=1, keepdims=True)
    col_best = np.min(clean, axis=0, keepdims=True)

    row_contrast = np.exp(-(clean - row_best) / scale)
    col_contrast = np.exp(-(clean - col_best) / scale)

    # Rank-based local preference, with a small temperature smoothing effect.
    row_order = np.argsort(np.argsort(clean, axis=1), axis=1).astype(float)
    col_order = np.argsort(np.argsort(clean, axis=0), axis=0).astype(float)

    temperature = max(1.0, np.sqrt(n))
    row_rank = np.exp(-row_order / temperature)
    col_rank = np.exp(-col_order / temperature)

    score = (
        0.30 * affinity
        + 0.25 * row_contrast
        + 0.25 * col_contrast
        + 0.20 * np.sqrt(row_rank * col_rank)
    )

    # Invalid edges and self-loops cannot be selected.
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    # Normalize each source's priors to make scores comparable across instances.
    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    return score.astype(float, copy=False)
