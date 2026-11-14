import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d += d.T
    d *= 0.5
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    work = np.where(valid, d, np.inf)

    kth = min(1, n - 2)
    local_scale = np.partition(work, kth=kth, axis=1)[:, kth]
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    work = np.where(
        np.isfinite(work),
        work,
        fallback_distance * 10.0,
    )
    np.maximum(work, eps, out=work)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scale_product = local_scale[:, None] * local_scale[None, :]
        np.sqrt(scale_product, out=scale_product)

        score = work / scale_product
        score *= -1.10
        np.exp(score, out=score)

        order = np.argsort(work, axis=1, kind="stable")
        rank = np.empty_like(score)
        rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
        rank /= max(n - 1, 1)

        rank += rank.T
        rank *= -3.20
        np.exp(rank, out=rank)
        score *= rank

        np.divide(work, global_scale, out=rank)
        rank *= -0.26
        np.exp(rank, out=rank)
        rank *= 0.38
        rank += 0.62
        score *= rank

        scale_difference = np.abs(
            local_scale[:, None] - local_scale[None, :]
        )
        scale_sum = local_scale[:, None] + local_scale[None, :]
        np.divide(
            scale_difference,
            scale_sum + eps,
            out=scale_difference,
        )
        scale_difference *= 0.85
        scale_difference += 1.0
        np.reciprocal(scale_difference, out=scale_difference)
        scale_difference *= 0.22
        scale_difference += 0.78
        score *= scale_difference

    score *= valid
    score += score.T
    score *= 0.5
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
