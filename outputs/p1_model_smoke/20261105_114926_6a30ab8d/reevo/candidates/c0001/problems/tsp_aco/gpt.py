import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]

    finite = np.isfinite(d) & (d > 0)
    safe_d = np.where(finite, d, np.inf)
    np.fill_diagonal(safe_d, np.inf)

    positive = safe_d[np.isfinite(safe_d)]
    scale = np.median(positive) if positive.size else 1.0

    inverse_distance = np.where(finite, 1.0 / (safe_d + 1e-12), 0.0)
    inverse_distance /= inverse_distance.max(initial=1.0)

    row_sorted = np.sort(safe_d, axis=1)
    local_scale = row_sorted[:, : min(5, max(1, n - 1))]
    local_scale = np.mean(
        np.where(np.isfinite(local_scale), local_scale, scale),
        axis=1,
    )
    local_scale = np.maximum(local_scale, 1e-12)

    mutual_proximity = np.exp(
        -safe_d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    )
    mutual_proximity = np.where(finite, mutual_proximity, 0.0)

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1)
    rank_score = np.where(
        finite,
        1.0 / (1.0 + ranks.astype(float)),
        0.0,
    )
    rank_score /= rank_score.max(initial=1.0)

    score = (
        0.50 * inverse_distance
        + 0.30 * mutual_proximity
        + 0.20 * rank_score
    )

    k = min(max(3, int(np.sqrt(max(n, 1)) + 1)), max(1, n - 1))
    keep = np.zeros_like(score, dtype=bool)
    for i in range(n):
        candidates = np.flatnonzero(finite[i])
        if candidates.size:
            selected = candidates[np.argsort(score[i, candidates])[-k:]]
            keep[i, selected] = True

    keep &= keep.T
    score = np.where(keep, score, 0.0)
    np.fill_diagonal(score, 0.0)

    row_max = score.max(axis=1, keepdims=True)
    score = np.divide(score, row_max, out=np.zeros_like(score), where=row_max > 0)

    return score.astype(float)
