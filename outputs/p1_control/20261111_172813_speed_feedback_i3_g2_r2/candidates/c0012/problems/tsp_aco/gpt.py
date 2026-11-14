import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    raw = distance_matrix.copy()
    np.fill_diagonal(raw, np.inf)

    finite = np.isfinite(raw)
    positive = finite & (raw > 0)

    if np.any(positive):
        fallback = float(np.median(raw[positive]))
    else:
        fallback = 1.0

    fallback = max(fallback, np.finfo(float).eps)
    distances = raw.copy()
    distances[~finite] = fallback
    distances = np.maximum(distances, np.finfo(float).eps)
    np.fill_diagonal(distances, np.inf)

    distances = 0.5 * (distances + distances.T)
    valid_edges = finite & finite.T
    np.fill_diagonal(valid_edges, False)

    inverse_distance = 1.0 / distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    inverse_distance[~valid_edges] = 0.0

    row_scale = np.max(inverse_distance, axis=1, keepdims=True)
    row_scale[row_scale <= 0] = 1.0
    local_preference = inverse_distance / row_scale

    competition = 1.0 / (
        np.sum(np.exp(-4.0 * local_preference), axis=1, keepdims=True)
        + 1e-12
    )
    competition_max = np.max(competition)
    if competition_max > 0:
        competition /= competition_max

    base_score = local_preference * np.sqrt(competition @ competition.T)

    nearest = np.argmin(
        np.where(valid_edges, distances, np.inf),
        axis=1,
    )
    has_neighbor = np.any(valid_edges, axis=1)

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if has_neighbor[i] and 0 <= j < n and has_neighbor[j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    mutual_nearest *= valid_edges

    alternative_distance = np.full(n, fallback, dtype=float)
    for i in range(n):
        candidates = distances[i].copy()
        candidates[i] = np.inf
        if has_neighbor[i]:
            alternative_distance[i] = np.min(candidates)

    savings = (
        alternative_distance[:, None]
        + alternative_distance[None, :]
        - 2.0 * distances
    )
    savings = np.maximum(savings, 0.0)
    savings[~valid_edges] = 0.0

    positive_savings = savings[savings > 0]
    if positive_savings.size:
        savings_scale = float(np.percentile(positive_savings, 90))
        savings_scale = max(savings_scale, np.finfo(float).eps)
        normalized_savings = np.minimum(savings / savings_scale, 1.0)
    else:
        normalized_savings = np.zeros_like(savings)

    score = (
        0.55 * base_score
        + 0.25 * mutual_nearest
        + 0.20 * normalized_savings
    )

    score = 0.5 * (score + score.T)
    score[~valid_edges] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.reshape(distance_matrix.shape)
