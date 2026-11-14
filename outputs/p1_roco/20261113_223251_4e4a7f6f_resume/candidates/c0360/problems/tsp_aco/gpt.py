import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized edge-prior matrix for stochastic TSP sampling."""
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

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    # Use a robust temperature with a conservative lower bound.
    temperature = max(0.58 * scale, eps)

    clean = distance.copy()
    finite_values = valid_values[np.isfinite(valid_values)]
    if finite_values.size:
        fallback = float(np.max(finite_values)) + 8.0 * scale
    else:
        fallback = 8.0 * scale
    clean[~valid] = fallback
    np.fill_diagonal(clean, fallback)

    # Main short-edge attraction. The additive form preserves alternatives.
    attraction = np.zeros((n, n), dtype=float)
    attraction[valid] = np.exp(
        -np.minimum(clean[valid] / temperature, 40.0)
    )

    # Stable endpoint-wise rank preference.
    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_width = max(1.0, 0.75 * np.sqrt(float(max(n - 1, 1))))
    rank_preference = np.exp(-ranks / rank_width)
    rank_preference[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_preference * rank_preference.T, 0.0)
    )

    # Reward edges that are especially important to at least both endpoints.
    nearest = np.min(clean, axis=1)
    if n >= 3:
        second_nearest = np.partition(clean, 1, axis=1)[:, 1]
    else:
        second_nearest = nearest + scale

    gap = np.maximum(second_nearest - nearest, 0.18 * scale)
    nearest_advantage = np.maximum(
        second_nearest[:, None] - clean,
        0.0,
    )
    endpoint_advantage = 1.0 - np.exp(
        -nearest_advantage / gap[:, None]
    )
    endpoint_advantage[~valid] = 0.0

    mutual_advantage = np.sqrt(
        np.maximum(
            endpoint_advantage * endpoint_advantage.T,
            0.0,
        )
    )

    # Mild degree correction prevents a dense hub from dominating.
    local_mass = attraction.sum(axis=1)
    degree_correction = 1.0 / np.sqrt(
        np.maximum(local_mass, eps)
    )
    degree_correction /= max(float(np.mean(degree_correction)), eps)
    degree_correction = np.clip(degree_correction, 0.72, 1.45)

    endpoint_correction = np.sqrt(
        np.maximum(
            degree_correction[:, None] * degree_correction[None, :],
            0.0,
        )
    )
    endpoint_correction[~valid] = 0.0

    score = (
        0.58 * attraction
        + 0.24 * mutual_rank
        + 0.14 * mutual_advantage
        + 0.04 * endpoint_correction * attraction
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    score /= maximum

    # One light balancing step reduces endpoint-scale bias without
    # washing out the local distance signal.
    row_mass = score.sum(axis=1)
    balancing = 1.0 / np.sqrt(np.maximum(row_mass, eps))
    balancing /= max(float(np.mean(balancing)), eps)
    balancing = np.clip(balancing, 0.80, 1.25)

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
