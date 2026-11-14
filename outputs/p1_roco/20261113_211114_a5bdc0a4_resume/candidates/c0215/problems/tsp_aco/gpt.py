import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (raw + raw.T)

    observed = symmetric[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.median(observed))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    finite_max = float(np.max(observed))
    replacement = finite_max + 16.0 * safe_scale

    distances = np.where(valid, symmetric, replacement)
    np.fill_diagonal(distances, replacement)

    normalized = np.minimum(distances / safe_scale, 60.0)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + safe_scale * 1e-9)

    inverse_max = max(float(np.max(inverse)), eps)
    inverse_relative = inverse / inverse_max

    temperature = max(0.72 * safe_scale, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / temperature, 60.0)
    )

    row_mass = np.maximum(affinity.sum(axis=1, keepdims=True), eps)
    col_mass = np.maximum(affinity.sum(axis=0, keepdims=True), eps)

    outgoing_share = affinity / row_mass
    incoming_share = affinity / col_mass
    mutual_competition = np.sqrt(
        np.maximum(outgoing_share * incoming_share, 0.0)
    )

    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    rank_support = np.exp(-ranks / rank_scale)
    rank_support[~valid] = 0.0

    neighborhood_size = min(
        n - 1,
        max(2, int(np.ceil(1.5 * np.sqrt(float(n))))),
    )

    neighborhood_support = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = order[i, : neighborhood_size + 1]
        neighbors = neighbors[neighbors != i][:neighborhood_size]

        if neighbors.size == 0:
            continue

        local_distances = distances[i, neighbors]
        local_scale = max(float(np.median(local_distances)), safe_scale)
        neighborhood_support[i, neighbors] = np.exp(
            -np.minimum(local_distances / local_scale, 40.0)
        )

    neighborhood_support = np.sqrt(
        np.maximum(
            neighborhood_support * neighborhood_support.T,
            0.0,
        )
    )

    degree = valid.sum(axis=1).astype(float)
    weighted_degree = np.maximum(affinity.sum(axis=1), eps)

    degree_scarcity = 1.0 / np.sqrt(
        np.maximum(degree, 1.0) * weighted_degree
    )
    degree_scarcity /= max(float(np.mean(degree_scarcity)), eps)
    degree_scarcity = np.clip(degree_scarcity, 0.65, 1.8)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            degree_scarcity[:, None] * degree_scarcity[None, :],
            0.0,
        )
    )

    distance_score = np.exp(-normalized)
    distance_score[~valid] = 0.0

    score = (
        0.42 * inverse_relative
        + 0.22 * mutual_competition
        + 0.16 * rank_support
        + 0.12 * neighborhood_support
        + 0.08 * endpoint_scarcity
    )

    score *= 0.78 + 0.22 * distance_score
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
    else:
        result = score / maximum

    exploration = float(
        np.clip(
            0.035 + 0.012 * np.std(normalized[valid]),
            0.035,
            0.09,
        )
    )

    valid_fraction = valid.astype(float)
    valid_count = max(float(np.sum(valid)), 1.0)
    uniform_prior = valid_fraction / valid_count

    result = (1.0 - exploration) * result + exploration * uniform_prior
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)

    for _ in range(12):
        row_sum = result.sum(axis=1, keepdims=True)
        result /= np.maximum(row_sum, eps)
        result = 0.5 * (result + result.T)

    result *= 0.5 + 0.5 * score
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
