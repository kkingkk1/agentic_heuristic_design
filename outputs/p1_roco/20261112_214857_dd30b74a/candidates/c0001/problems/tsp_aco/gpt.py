import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    x = d.copy()
    finite_offdiag = np.isfinite(x)
    np.fill_diagonal(finite_offdiag, False)

    valid_values = x[finite_offdiag]
    if valid_values.size == 0:
        return np.zeros_like(x, dtype=float)

    positive_values = valid_values[valid_values > 0]
    fallback = float(np.median(positive_values)) if positive_values.size else 1.0
    x[~np.isfinite(x)] = fallback * 1e6
    x[x < 0] = fallback * 1e6
    np.fill_diagonal(x, np.inf)

    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    ordered = np.sort(x, axis=1)
    local_scale = np.median(ordered[:, :k], axis=1)
    local_scale[~np.isfinite(local_scale) | (local_scale <= 0)] = fallback

    scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    scale[scale <= 0] = fallback

    relative_shortness = np.exp(-x / (scale + 1e-12))

    ranks = np.argsort(np.argsort(x, axis=1), axis=1).astype(float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_consensus = np.sqrt(rank_score * rank_score.T)

    row_medians = np.median(ordered[:, :max(1, min(n - 1, 2 * k))], axis=1)
    row_medians[row_medians <= 0] = fallback
    scale_match = np.exp(
        -np.abs(np.log((x + 1e-12) / (row_medians[:, None] + 1e-12)))
    )
    scale_match *= np.exp(
        -np.abs(np.log((x + 1e-12) / (row_medians[None, :] + 1e-12)))
    )

    score = (
        0.55 * relative_shortness
        + 0.30 * rank_consensus
        + 0.15 * scale_match
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
