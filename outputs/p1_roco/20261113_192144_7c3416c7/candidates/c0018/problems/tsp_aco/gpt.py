import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    score = np.zeros_like(distance_matrix, dtype=float)

    if n < 2:
        return score

    d = np.array(distance_matrix, dtype=float, copy=True)
    d[~np.isfinite(d)] = np.nan
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.nan)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return score

    positive_values = finite_values[finite_values > 0]
    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0
    scale = max(scale, np.finfo(float).eps)

    fallback = float(np.nanmax(d)) if np.isfinite(d).any() else scale
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = scale

    safe_d = np.where(np.isfinite(d), d, fallback * 4.0)

    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values[None, :]
    rank_score = 1.0 / (1.0 + ranks)

    nearest = order[:, 0]
    mutual = (nearest[:, None] == np.arange(n)[None, :]) & (
        nearest[None, :] == np.arange(n)[:, None]
    )
    mutual_bonus = mutual.astype(float)

    sorted_distances = np.take_along_axis(safe_d, order, axis=1)
    first = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else first + scale
    regret = np.maximum(second[:, None] - safe_d, 0.0) / scale
    regret = np.minimum(regret, 5.0)

    distance_term = np.exp(-safe_d / scale)
    rank_term = np.sqrt(rank_score * rank_score.T)
    regret_term = 1.0 + 0.35 * np.sqrt(regret * regret.T)
    mutual_term = 1.0 + 0.75 * mutual_bonus

    score = distance_term * (0.55 + 0.45 * rank_term) * regret_term * mutual_term
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.reshape(distance_matrix.shape)
