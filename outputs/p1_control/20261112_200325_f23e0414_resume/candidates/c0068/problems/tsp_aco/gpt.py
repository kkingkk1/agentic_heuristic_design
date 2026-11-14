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

    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    inv_distance = 1.0 / (d + eps)
    inv_distance[~np.isfinite(inv_distance)] = 0.0

    if n > 2:
        sorted_distances = np.sort(d, axis=1)
        nearest = sorted_distances[:, 0]
        second_nearest = sorted_distances[:, 1]

        regret = np.maximum(second_nearest - nearest, 0.0)
        positive_regret = regret[regret > 0]
        regret_scale = (
            float(np.median(positive_regret))
            if positive_regret.size
            else scale
        )

        regret_ratio = regret / (regret_scale + eps)
        regret_weight = 1.0 + np.sqrt(np.minimum(regret_ratio, 9.0))

        row_preference = inv_distance / (
            np.sum(inv_distance, axis=1, keepdims=True) + eps
        )

        mutual_preference = np.sqrt(
            np.maximum(row_preference, 0.0)
            * np.maximum(row_preference.T, 0.0)
        )

        score = inv_distance * (0.4 + 0.8 * mutual_preference)
        score *= np.sqrt(
            regret_weight[:, None] * regret_weight[None, :]
        )
    else:
        score = inv_distance

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score) if score.size else 0.0
    if not np.isfinite(maximum) or maximum <= 0:
        score = np.ones((n, n), dtype=float)
        np.fill_diagonal(score, 0.0)
    else:
        score /= maximum

    score += 1e-12
    np.fill_diagonal(score, 0.0)

    return score
