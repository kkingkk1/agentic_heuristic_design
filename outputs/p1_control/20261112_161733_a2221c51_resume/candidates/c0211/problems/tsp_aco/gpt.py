import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, degree-balanced stochastic edge-prior matrix."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    d = np.where(finite, d, np.inf)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid]

    median = float(np.median(positive))
    deviation = float(np.median(np.abs(positive - median)))
    temperature = max(median + deviation, eps)

    scaled = np.divide(
        d,
        temperature,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        kernel = np.exp(-np.minimum(scaled, 745.0))

    kernel[~valid] = 0.0
    np.fill_diagonal(kernel, 0.0)

    degree = np.sum(valid, axis=1).astype(float)
    target = np.minimum(2.0, degree)
    active = target > 0.0

    scaling = np.ones(n, dtype=float)
    scaling[~active] = 0.0

    for _ in range(96):
        balanced = scaling[:, None] * kernel * scaling[None, :]
        row_sum = np.sum(balanced, axis=1)

        ratio = np.ones(n, dtype=float)
        usable = active & (row_sum > eps) & np.isfinite(row_sum)
        ratio[usable] = np.sqrt(
            np.clip(target[usable] / row_sum[usable], 0.0, 1e150)
        )
        ratio[~active] = 0.0

        new_scaling = scaling * ratio
        new_scaling = np.clip(new_scaling, 0.0, 1e150)

        if np.max(np.abs(new_scaling - scaling)) <= 1e-10:
            scaling = new_scaling
            break

        scaling = new_scaling

    score = scaling[:, None] * kernel * scaling[None, :]
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
