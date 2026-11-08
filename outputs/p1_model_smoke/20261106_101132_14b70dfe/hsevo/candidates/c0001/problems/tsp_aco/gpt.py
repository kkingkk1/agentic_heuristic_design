import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    eps = np.finfo(float).eps

    valid = np.isfinite(d) & (d > 0)
    masked = np.where(valid, d, np.inf)

    scale = np.median(masked[np.isfinite(masked)]) if np.any(np.isfinite(masked)) else 1.0
    scale = max(scale, eps)

    inverse_cost = np.where(valid, 1.0 / (d + eps), 0.0)

    local_mean = np.mean(
        np.where(valid, d, np.nan),
        axis=1,
        where=valid
    )
    local_mean = np.where(np.isfinite(local_mean), local_mean, scale)
    scarcity = 1.0 / (local_mean + eps)

    endpoint_quality = np.sqrt(scarcity[:, None] * scarcity[None, :])
    normalized_cost = d / scale
    proximity = np.exp(-np.clip(normalized_cost, 0.0, 50.0))

    rank_score = np.zeros_like(d)
    for i in range(n):
        order = np.argsort(masked[i])
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        rank_score[i] = np.where(valid[i], 1.0 / (ranks + 1.0), 0.0)

    score = (
        inverse_cost
        * (0.55 + 0.25 * proximity + 0.20 * rank_score)
        * (1.0 + endpoint_quality / (endpoint_quality.max() + eps))
    )

    np.fill_diagonal(score, 0.0)
    score[~valid] = 0.0

    return score
