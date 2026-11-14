import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    finite = np.isfinite(d)
    valid = finite & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    safe_d = np.where(valid, d, 0.0)

    counts = np.sum(valid, axis=1)
    counts = np.maximum(counts, 1)

    rms_scale = np.sqrt(
        np.sum(safe_d * safe_d, axis=1) / counts
    )
    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    rms_scale = np.maximum(rms_scale, max(fallback, eps))

    scale = rms_scale[:, None]
    normalized = safe_d / scale

    affinity = np.exp(-0.5 * normalized * normalized)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_mass = np.sum(affinity, axis=1, keepdims=True)
    transition = affinity / np.maximum(row_mass, eps)

    mutual_affinity = np.sqrt(
        np.maximum(transition, 0.0)
        * np.maximum(transition.T, 0.0)
    )

    neighborhood_overlap = affinity @ affinity.T
    neighborhood_norm = np.sqrt(
        np.maximum(
            np.diag(neighborhood_overlap)[:, None]
            * np.diag(neighborhood_overlap)[None, :],
            eps,
        )
    )
    overlap_ratio = neighborhood_overlap / neighborhood_norm

    exclusivity = 1.0 / (1.0 + np.maximum(overlap_ratio, 0.0))
    score = mutual_affinity * exclusivity

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))

    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
