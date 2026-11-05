import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square NumPy array")

    n = d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    work = d.copy()
    work[~np.isfinite(work)] = np.inf
    np.fill_diagonal(work, np.inf)

    finite_values = work[np.isfinite(work)]
    if finite_values.size == 0:
        return np.zeros_like(work, dtype=float)

    fallback = float(np.max(finite_values))
    work[~np.isfinite(work)] = fallback

    row_min = np.min(work, axis=1, keepdims=True)
    row_second = np.partition(work, 1, axis=1)[:, 1:2]
    regret = (row_second - work) / (row_second + eps)
    regret = np.maximum(regret, 0.0)

    inverse = 1.0 / (work + eps)
    inverse /= np.max(inverse, axis=1, keepdims=True) + eps

    two_hop = np.min(work[:, :, None] + work[None, :, :], axis=1)
    bridge = (two_hop - work) / (two_hop + work + eps)
    bridge = np.maximum(bridge, 0.0)

    score = 0.55 * inverse + 0.30 * regret + 0.15 * bridge
    score = 0.5 * (score + score.T)

    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    return score
