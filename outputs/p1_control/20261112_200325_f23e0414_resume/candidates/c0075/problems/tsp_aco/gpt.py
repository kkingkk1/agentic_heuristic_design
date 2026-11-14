import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = d / (pair_scale + eps)

    closeness = np.exp(-relative_length)

    competitiveness = 1.0 / (1.0 + relative_length)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    log_scale = np.log(local_scale)
    scale_gap = np.abs(
        log_scale[:, None] - log_scale[None, :]
    )

    off_diagonal = ~np.eye(n, dtype=bool)
    gap_values = scale_gap[off_diagonal]

    if gap_values.size:
        upper_quartile = float(np.quantile(gap_values, 0.75))
        bridge_mask = scale_gap >= upper_quartile
    else:
        bridge_mask = np.zeros_like(scale_gap, dtype=bool)

    score = (
        closeness
        * (0.60 + 0.40 * mutual_competitiveness)
        * (1.0 + 0.08 * bridge_mask.astype(float))
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    lo = float(np.min(values))
    hi = float(np.max(values))

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
