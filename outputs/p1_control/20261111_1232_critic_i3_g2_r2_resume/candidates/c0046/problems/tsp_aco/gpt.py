import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    diagonal = np.eye(n, dtype=bool)

    if np.any(~finite & ~diagonal):
        raise ValueError(
            "distance_matrix contains invalid non-finite off-diagonal values"
        )

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(1.0, fallback)

    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    nearest_idx = np.argpartition(work, kth=k - 1, axis=1)[:, :k]
    nearest_values = np.take_along_axis(work, nearest_idx, axis=1)

    local_scale = np.median(nearest_values, axis=1)
    valid_scale = np.isfinite(local_scale) & (local_scale > 0.0)
    scale_fallback = (
        float(np.median(local_scale[valid_scale]))
        if np.any(valid_scale)
        else fallback
    )
    local_scale = np.where(
        np.isfinite(local_scale),
        np.maximum(local_scale, eps),
        max(scale_fallback, eps),
    )

    candidate = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate[rows, nearest_idx] = True

    mutual = candidate & candidate.T
    candidate |= mutual

    endpoint_scale = local_scale[:, None] + local_scale[None, :]
    normalized = work / (endpoint_scale + eps)

    local_score = np.exp(-1.15 * normalized)
    local_score[~finite] = 0.0
    local_score[diagonal] = 0.0

    density = local_scale / (np.median(local_scale) + eps)
    isolation = np.clip(density, 0.5, 3.0)
    isolation /= max(float(np.max(isolation)), 1.0)

    bridge_bonus = 1.0 + 0.22 * np.maximum(
        isolation[:, None],
        isolation[None, :],
    ) * np.exp(-normalized)

    score = local_score * bridge_bonus

    finite_offdiag = finite & ~diagonal
    candidate_strength = np.where(candidate, 1.0, 0.08)
    score *= candidate_strength

    if np.any(finite_offdiag):
        finite_scores = score[finite_offdiag]
        positive_scores = finite_scores[finite_scores > 0.0]
        if positive_scores.size:
            baseline = 0.02 * float(np.max(positive_scores))
            score[finite_offdiag & ~candidate] = np.maximum(
                score[finite_offdiag & ~candidate],
                baseline,
            )

    score[~finite] = 0.0
    score[diagonal] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
