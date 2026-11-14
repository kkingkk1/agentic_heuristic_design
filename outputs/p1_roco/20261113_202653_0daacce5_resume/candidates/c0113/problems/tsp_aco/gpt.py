import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be nonnegative")

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    positive = d[valid & (d > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(global_scale, 1.0)

    row_scales = np.full(n, global_scale, dtype=float)
    for i in range(n):
        values = d[i, valid[i] & (d[i] > 0)]
        if values.size:
            row_scales[i] = float(np.median(values))

    row_scales = np.maximum(row_scales, eps)
    blended_scale = 0.7 * row_scales + 0.3 * global_scale

    local_exp = np.zeros_like(d)
    local_exp[valid] = np.exp(
        -d[valid] / np.repeat(blended_scale, n).reshape(n, n)[valid]
    )

    inverse = np.zeros_like(d)
    inverse[valid] = 1.0 / np.maximum(d[valid], eps)

    local_exp /= np.maximum(local_exp.sum(axis=1, keepdims=True), eps)
    inverse /= np.maximum(inverse.sum(axis=1, keepdims=True), eps)

    mutual_exp = np.sqrt(local_exp * local_exp.T)
    mutual_inverse = np.sqrt(inverse * inverse.T)

    ranks = np.zeros_like(d)
    for i in range(n):
        order = np.argsort(d[i], kind="stable")
        rank = 0
        for j in order:
            if i != j and np.isfinite(d[i, j]):
                ranks[i, j] = 1.0 / (1.0 + rank)
                rank += 1

    mutual_rank = np.sqrt(ranks * ranks.T)

    row_best = np.max(
        np.where(valid, 0.65 * local_exp + 0.35 * inverse, 0.0),
        axis=1,
        keepdims=True,
    )
    relative_strength = np.zeros_like(d)
    base_strength = 0.65 * local_exp + 0.35 * inverse
    relative_strength[valid] = (
        base_strength[valid]
        / np.maximum(
            np.repeat(row_best, n, axis=1)[valid],
            eps,
        )
    )

    score = (
        0.40 * mutual_exp
        + 0.30 * mutual_inverse
        + 0.20 * mutual_rank
        + 0.10 * np.sqrt(relative_strength * relative_strength.T)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
