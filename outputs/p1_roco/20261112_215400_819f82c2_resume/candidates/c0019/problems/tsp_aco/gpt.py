import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    work = distances.copy()
    diagonal = np.eye(n, dtype=bool)

    finite_values = work[~diagonal & np.isfinite(work)]
    if finite_values.size == 0:
        return np.zeros_like(work, dtype=float)

    fallback = float(np.max(finite_values))
    work[~np.isfinite(work)] = fallback
    work = np.maximum(work, 0.0)
    work[diagonal] = np.inf

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for row in range(n):
        ranks[row, order[row]] = rank_values

    rank_affinity = 1.0 / (1.0 + ranks)

    positive = work[np.isfinite(work) & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    distance_affinity = np.exp(-work / scale)
    distance_affinity[diagonal] = 0.0

    mutual_affinity = np.sqrt(rank_affinity * rank_affinity.T)
    local_affinity = 0.5 * (rank_affinity + rank_affinity.T)

    scores = (
        0.55 * distance_affinity
        + 0.30 * mutual_affinity
        + 0.15 * local_affinity
    )

    scores[diagonal] = 0.0
    scores = 0.5 * (scores + scores.T)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
