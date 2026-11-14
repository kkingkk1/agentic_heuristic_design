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

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    if not np.any(positive):
        return np.zeros_like(d, dtype=float)

    fallback = np.max(d[positive])
    fallback = fallback if np.isfinite(fallback) and fallback > 0 else 1.0

    clean = np.where(positive, d, fallback)
    np.fill_diagonal(clean, fallback)

    eps = np.finfo(float).eps
    inv = 1.0 / np.maximum(clean, eps)

    def row_normalize(x):
        lo = np.min(x, axis=1, keepdims=True)
        hi = np.max(x, axis=1, keepdims=True)
        return (x - lo) / (hi - lo + eps)

    direct = row_normalize(inv)

    # Prefer edges that are locally competitive at both endpoints.
    row_rank = inv / (np.sum(inv, axis=1, keepdims=True) + eps)
    competition = np.sqrt(row_rank * row_rank.T)

    # Two-step lookahead: endpoints with good access to nearby alternatives
    # are less likely to create premature dead ends.
    neighborhood = inv.copy()
    np.fill_diagonal(neighborhood, 0.0)
    lookahead = neighborhood @ neighborhood
    lookahead = row_normalize(lookahead)

    score = (
        0.55 * direct
        + 0.30 * competition
        + 0.15 * lookahead
    )

    score = 0.5 * (score + score.T)
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    score -= np.min(score)
    maximum = np.max(score)
    if maximum > eps:
        score /= maximum

    return np.asarray(score, dtype=float)
