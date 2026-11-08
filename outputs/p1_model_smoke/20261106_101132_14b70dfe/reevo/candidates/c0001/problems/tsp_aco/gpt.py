import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    eps = np.finfo(float).eps

    valid = np.isfinite(d) & (d > 0)
    safe_d = np.where(valid, d, np.inf)

    local_scale = np.nanmedian(
        np.where(valid, d, np.nan),
        axis=1,
        keepdims=True,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        np.nanmedian(np.where(valid, d, np.nan)),
    )
    local_scale = np.maximum(local_scale, eps)

    attractiveness = 1.0 / np.maximum(safe_d, eps)
    relative_closeness = 1.0 / (safe_d / local_scale + eps)

    nearest = np.min(safe_d, axis=1, keepdims=True)
    nearest = np.maximum(nearest, eps)
    regret = np.maximum(safe_d - nearest, 0.0) / (local_scale + eps)
    regret_bonus = np.exp(-regret)

    rank_score = np.zeros_like(d)
    for i in range(n):
        order = np.argsort(safe_d[i])
        rank_score[i, order] = np.arange(n, dtype=float)
    rank_score = np.exp(-rank_score / max(2.0, np.sqrt(max(n, 1))))

    score = (
        attractiveness ** 0.55
        * relative_closeness ** 0.25
        * regret_bonus ** 0.35
        * (0.5 + 0.5 * rank_score)
    )

    candidate_count = max(2, min(n - 1, int(np.ceil(np.sqrt(max(n, 1))) + 2)))
    keep = np.zeros_like(valid)
    for i in range(n):
        candidates = np.argsort(safe_d[i])[:candidate_count]
        keep[i, candidates] = True

    mutual = keep & keep.T
    score *= np.where(mutual, 1.35, np.where(keep, 0.75, 0.12))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    max_score = np.max(score, axis=1, keepdims=True)
    score = np.divide(score, max_score, out=np.zeros_like(score), where=max_score > 0)
    return score
