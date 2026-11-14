import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    finite_values = d[np.isfinite(d) & (d >= 0)]
    if finite_values.size == 0:
        raise ValueError("distance_matrix contains no valid distances")

    fallback = float(np.max(finite_values)) + 1.0
    d[~np.isfinite(d)] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    order = np.argsort(d, axis=1)
    rank = np.empty_like(order, dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    rank = rank / max(n - 1, 1)

    k_values = sorted(set([
        max(1, int(np.sqrt(n))),
        max(1, n // 4),
        max(1, n // 2),
    ]))

    score = np.zeros((n, n), dtype=float)

    for k in k_values:
        nearest = np.take_along_axis(d, order[:, :k], axis=1)
        local_scale = np.median(nearest, axis=1)
        local_scale = np.maximum(local_scale, global_scale * 1e-6)

        directed_affinity = np.exp(
            -d / local_scale[:, None]
        )
        directed_affinity[~np.isfinite(directed_affinity)] = 0.0

        mutual_affinity = np.sqrt(
            directed_affinity * directed_affinity.T
        )

        rank_consensus = np.exp(
            -3.0 * np.sqrt(np.maximum(rank * rank.T, 0.0))
        )

        distance_preference = np.exp(-d / global_scale)
        distance_preference[~np.isfinite(distance_preference)] = 0.0

        score += (
            0.45 * mutual_affinity
            + 0.35 * rank_consensus
            + 0.20 * distance_preference
        )

    score /= float(len(k_values))
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float)
