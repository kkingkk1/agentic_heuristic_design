import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid_input = np.isfinite(distances) & (distances > 0)
    finite_values = distances[valid_input]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(finite_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    d = np.where(valid_input, distances, fallback)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_distances = d[valid]
    global_scale = float(np.median(finite_distances))
    global_scale = max(global_scale, np.finfo(float).eps)

    order = np.argsort(d, axis=1)
    nearest = order[:, 0]
    second_nearest = order[:, 1] if n > 2 else nearest

    nearest_distance = d[np.arange(n), nearest]
    second_distance = d[np.arange(n), second_nearest]

    regret = np.zeros(n, dtype=float)
    if n > 2:
        regret = (second_distance - nearest_distance) / np.maximum(
            second_distance,
            np.finfo(float).eps,
        )
        regret = np.clip(regret, 0.0, 1.0)

    mutual = (
        nearest[:, None] == np.arange(n)[None, :]
    ) & (
        nearest[None, :] == np.arange(n)[:, None]
    )

    relative_distance = d / global_scale
    distance_score = 1.0 / np.maximum(relative_distance, np.finfo(float).eps)

    rank = np.empty_like(order, dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_bonus = 1.0 + 0.10 * np.exp(
        -rank / max(1.0, np.sqrt(float(n)))
    )

    endpoint_bonus = 1.0 + 0.12 * (
        regret[:, None] + regret[None, :]
    ) * 0.5

    score = distance_score * rank_bonus * endpoint_bonus
    score *= np.where(mutual, 1.30, 1.0)
    score *= valid

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
