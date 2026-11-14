import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(finite_positive)) if finite_positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Distance attractiveness: shorter edges receive larger values.
    distance_score = np.exp(-d / scale)
    distance_score[~np.isfinite(distance_score)] = 0.0

    if n > 1:
        order = np.argsort(d, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=float)
        ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
        ranks = np.maximum(ranks - 1.0, 0.0)

        # Rank attractiveness emphasizes each node's nearest neighbors.
        rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))

        # Mutual-neighborhood reinforcement favors edges preferred by both endpoints.
        mutual_score = np.sqrt(rank_score * rank_score.T)

        score = (
            0.55 * distance_score
            + 0.25 * rank_score
            + 0.20 * mutual_score
        )
    else:
        score = np.zeros_like(d)

    score = np.maximum(score, score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score
