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

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]

    eps = np.finfo(float).eps
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = d / (pair_scale + eps)

    closeness = np.exp(-relative)

    competitiveness = 1.0 / (1.0 + relative)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    scale_ratio = local_scale[:, None] / local_scale[None, :]
    reciprocal_ratio = np.maximum(
        scale_ratio,
        1.0 / np.maximum(scale_ratio, eps),
    )
    bridge = reciprocal_ratio >= 2.5

    score = (
        closeness
        * (0.65 + 0.35 * mutual_competitiveness)
        * (1.0 + 0.10 * bridge.astype(float))
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
