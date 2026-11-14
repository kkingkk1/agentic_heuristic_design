import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-prior matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    distance = np.full((n, n), np.inf, dtype=float)
    distance[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distance, np.inf)

    valid_values = distance[valid]
    positive_values = valid_values[valid_values > 0.0]

    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    finite_values = valid_values[np.isfinite(valid_values)]
    fallback = (
        float(np.max(finite_values)) + 8.0 * scale
        if finite_values.size
        else 8.0 * scale
    )

    clean = distance.copy()
    clean[~valid] = fallback
    np.fill_diagonal(clean, fallback)

    temperature = max(0.58 * scale, eps)

    attraction = np.zeros((n, n), dtype=float)
    attraction[valid] = np.exp(
        -np.minimum(clean[valid] / temperature, 40.0)
    )

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_width = max(1.0, 0.75 * np.sqrt(float(max(n - 1, 1))))
    rank_preference = np.exp(-ranks / rank_width)
    rank_preference[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_preference * rank_preference.T, 0.0)
    )

    nearest = np.min(clean, axis=1)

    if n >= 3:
        second_nearest = np.partition(clean, 1, axis=1)[:, 1]
    else:
        second_nearest = nearest + scale

    local_gap = np.maximum(
        second_nearest - nearest,
        0.18 * scale,
    )

    nearest_advantage = np.maximum(
        second_nearest[:, None] - clean,
        0.0,
    )

    endpoint_advantage = 1.0 - np.exp(
        -nearest_advantage / local_gap[:, None]
    )
    endpoint_advantage[~valid] = 0.0

    mutual_advantage = np.sqrt(
        np.maximum(
            endpoint_advantage * endpoint_advantage.T,
            0.0,
        )
    )

    # Weak bridge term: preserve edges that are highly competitive
    # for at least one endpoint, which helps sparse or boundary nodes.
    unilateral_rank = np.maximum(
        rank_preference,
        rank_preference.T,
    )
    unilateral_rank[~valid] = 0.0

    # Mild endpoint correction prevents dense regions from dominating.
    local_mass = attraction.sum(axis=1)
    endpoint_correction = 1.0 / np.sqrt(
        np.maximum(local_mass, eps)
    )
    endpoint_correction /= max(
        float(np.mean(endpoint_correction)),
        eps,
    )
    endpoint_correction = np.clip(
        endpoint_correction,
        0.78,
        1.35,
    )

    correction_matrix = np.sqrt(
        np.maximum(
            endpoint_correction[:, None]
            * endpoint_correction[None, :],
            0.0,
        )
    )
    correction_matrix[~valid] = 0.0

    score = (
        0.57 * attraction
        + 0.23 * mutual_rank
        + 0.14 * mutual_advantage
        + 0.04 * unilateral_rank
    )

    score *= 0.96 + 0.04 * correction_matrix
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    score /= maximum

    # One light balancing step only; avoid iterative global reshaping.
    row_mass = score.sum(axis=1)
    balancing = 1.0 / np.sqrt(
        np.maximum(row_mass, eps)
    )
    balancing /= max(float(np.mean(balancing)), eps)
    balancing = np.clip(balancing, 0.82, 1.22)

    score *= np.sqrt(
        np.maximum(
            balancing[:, None] * balancing[None, :],
            0.0,
        )
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    result = score / maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
