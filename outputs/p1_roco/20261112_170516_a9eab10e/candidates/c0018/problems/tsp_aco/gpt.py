import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    finite_off_diag = np.isfinite(d) & ~np.eye(n, dtype=bool)

    if not np.any(finite_off_diag):
        return np.zeros((n, n), dtype=float)

    positive = d[finite_off_diag & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0
    d[~np.isfinite(d)] = fallback * 10.0
    d[d < 0] = fallback * 10.0

    np.fill_diagonal(d, np.inf)

    sorted_d = np.sort(d, axis=1)
    nearest = sorted_d[:, 0]
    second_nearest = sorted_d[:, 1] if n > 2 else nearest.copy()

    scale = np.maximum(second_nearest, fallback * 1e-12)
    inverse_distance = 1.0 / np.maximum(d, fallback * 1e-12)
    inverse_distance /= np.maximum(np.max(inverse_distance, axis=1, keepdims=True), 1e-12)

    rank_signal = np.maximum(0.0, 1.0 - (np.argsort(np.argsort(d, axis=1), axis=1) / max(n - 1, 1)))
    rank_signal[np.eye(n, dtype=bool)] = 0.0

    regret_u = np.maximum(second_nearest - nearest, 0.0) / scale
    regret_v = regret_u.copy()
    regret_signal = (regret_u[:, None] + regret_v[None, :]) / 2.0
    regret_signal /= max(float(np.max(regret_signal)), 1e-12)

    mutual = np.zeros((n, n), dtype=float)
    nearest_count = min(3, n - 1)
    for i in range(n):
        neighbors = np.argsort(d[i])[:nearest_count]
        mutual[i, neighbors] = 1.0
    mutual = np.maximum(mutual, mutual.T)

    score = (
        0.50 * inverse_distance
        + 0.25 * rank_signal
        + 0.15 * regret_signal
        + 0.10 * mutual
    )

    score = (score + score.T) / 2.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = score / np.maximum(row_max, 1e-12)
    score[~finite_off_diag] = 0.0
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
