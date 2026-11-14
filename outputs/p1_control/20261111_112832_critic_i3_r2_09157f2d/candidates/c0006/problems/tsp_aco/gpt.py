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

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = d[finite_nonnegative]

    if valid.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(valid))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    clean = np.where(finite_nonnegative, d, fallback)
    np.fill_diagonal(clean, np.inf)

    finite_edges = clean[np.isfinite(clean)]
    positive_edges = finite_edges[finite_edges > 0.0]

    if positive_edges.size:
        scale = float(np.median(positive_edges))
    elif finite_edges.size:
        scale = float(np.mean(finite_edges))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_clean = np.where(np.isfinite(clean), clean, fallback)

    inverse = 1.0 / (finite_clean + 1e-12)
    inverse = inverse * scale
    inverse = np.nan_to_num(
        inverse,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    shortness = np.exp(-finite_clean / scale)
    shortness = np.nan_to_num(shortness, nan=0.0, posinf=1.0, neginf=0.0)

    ranks = np.zeros((n, n), dtype=float)
    nearest = np.full(n, -1, dtype=int)
    second_nearest = np.full(n, fallback, dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(np.arange(n) != i)
        ordered = candidates[np.argsort(finite_clean[i, candidates], kind="stable")]

        if ordered.size:
            nearest[i] = int(ordered[0])
            if ordered.size > 1:
                second_nearest[i] = finite_clean[i, ordered[1]]

            denominator = max(ordered.size - 1, 1)
            ranks[i, ordered] = 1.0 - (
                np.arange(ordered.size, dtype=float) / denominator
            )

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    nearest_distance = np.min(finite_clean, axis=1)
    gaps = np.maximum(second_nearest - nearest_distance, 0.0)

    competition = np.zeros((n, n), dtype=float)
    for i in range(n):
        valid = np.arange(n) != i
        denominator = gaps[i] + 0.1 * scale
        competition[i, valid] = np.exp(
            -(finite_clean[i, valid] - nearest_distance[i])
            / max(denominator, 1e-12)
        )

    score = (
        0.35 * inverse
        + 0.25 * shortness
        + 0.20 * ranks
        + 0.10 * competition
        + 0.10 * mutual
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    return score
