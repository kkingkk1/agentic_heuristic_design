import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, d, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe)
    np.fill_diagonal(valid, False)

    finite = safe[valid]
    positive = finite[finite > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.median(finite)) if finite.size else 1.0
    scale = max(scale, eps)

    normalized = np.zeros((n, n), dtype=float)
    normalized[valid] = safe[valid] / scale

    # A bounded global kernel with a soft transition around the typical edge length.
    direct = np.zeros((n, n), dtype=float)
    direct[valid] = 1.0 / (1.0 + normalized[valid] ** 1.7)

    # Row-normalized transition probabilities over usable neighbors.
    row_sum = direct.sum(axis=1, keepdims=True)
    transition = np.divide(
        direct,
        row_sum,
        out=np.zeros_like(direct),
        where=row_sum > 0.0,
    )

    # Two-hop compatibility: edges are favored when their endpoints share
    # high-probability intermediate neighbors.
    closure = transition @ transition.T
    closure = np.sqrt(np.maximum(closure, 0.0))

    # Prefer edges that are useful to both endpoint transition distributions.
    endpoint_mass = np.sqrt(
        np.maximum(row_sum @ row_sum.T, 0.0)
    )
    endpoint_mass /= endpoint_mass.max() if endpoint_mass.max() > 0.0 else 1.0

    score = (
        0.52 * direct
        + 0.38 * closure
        + 0.10 * endpoint_mass
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
