import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(d) & off_diagonal
    valid = finite | finite.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[finite & (d > 0)]
    if positive.size:
        fallback = float(np.median(positive))
    else:
        fallback = 1.0

    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.where(np.isfinite(d), d, fallback)
    clean = np.maximum(clean, 0.0)
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, 0.0)

    positive_clean = clean[valid & (clean > 0)]
    if positive_clean.size:
        temperature = float(np.median(positive_clean))
    else:
        temperature = fallback

    temperature = max(temperature, np.finfo(float).eps)

    affinity = np.exp(
        -np.clip(clean / temperature, 0.0, 60.0)
    )
    affinity *= valid
    np.fill_diagonal(affinity, 0.0)

    for _ in range(100):
        row_sum = affinity.sum(axis=1)
        factors = np.ones(n, dtype=float)

        active = row_sum > 0
        factors[active] = np.sqrt(2.0 / row_sum[active])
        factors = np.clip(factors, 1e-6, 1e6)

        affinity *= factors[:, None] * factors[None, :]
        affinity *= valid
        np.fill_diagonal(affinity, 0.0)

    row_sum = affinity.sum(axis=1, keepdims=True)
    normalized = np.divide(
        affinity,
        row_sum,
        out=np.zeros_like(affinity),
        where=row_sum > 0,
    )

    score = 0.5 * (normalized + normalized.T)
    score *= valid
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
