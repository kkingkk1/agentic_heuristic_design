import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(d)
    valid_values = d[finite & (d >= 0)]

    if valid_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(valid_values)) if valid_values.size else 1.0
    if fallback <= 0:
        fallback = 1.0

    clean = np.where(finite & (d >= 0), d, fallback)
    np.fill_diagonal(clean, np.inf)

    positive = clean[np.isfinite(clean) & (clean > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0

    inverse = 1.0 / (clean + 1e-12)
    inverse[~np.isfinite(inverse)] = 0.0

    ranks = np.argsort(np.argsort(clean, axis=1), axis=1).astype(float)
    ranks = np.maximum(ranks, 0.0)
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))

    score = 0.65 * (inverse * scale) + 0.35 * rank_score
    score = 0.5 * (score + score.T)

    finite_score = score[np.isfinite(score)]
    if finite_score.size:
        upper = np.percentile(finite_score, 99)
        if upper > 0 and np.isfinite(upper):
            score = np.clip(score / upper, 0.0, 1.0)

    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
