import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps
    finite_positive = np.isfinite(d) & (d > 0)
    valid = finite_positive & finite_positive.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[finite_positive]
    fallback = float(np.max(positive_values))
    replacement = max(fallback * 10.0, eps)

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    kth = min(1, n - 1)
    local_scale = np.maximum(
        np.partition(work, kth=kth, axis=1)[:, kth],
        eps,
    )

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = work / endpoint_scale

    global_scale = max(float(np.median(positive_values)), eps)
    absolute_length = work / global_scale

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )

    # Partial selection preserves promising edges without a full sort.
    top_k = min(8, n - 1)
    top_indices = np.argpartition(work, kth=top_k - 1, axis=1)[:, :top_k]
    top_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    top_mask[rows, top_indices] = True
    top_mask &= valid

    mutual_top = top_mask & top_mask.T
    one_sided_top = top_mask | top_mask.T

    bonus = np.ones((n, n), dtype=float)
    bonus[one_sided_top] = 1.12
    bonus[mutual_top] = 1.35

    denominator = (
        1.0
        + 0.90 * normalized_length
        + 0.25 * absolute_length
        + 0.40 * scale_difference
    )
    score = bonus / denominator

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
