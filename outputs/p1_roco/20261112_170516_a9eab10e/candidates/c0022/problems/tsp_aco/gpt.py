import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    valid_offdiag = finite.copy()
    np.fill_diagonal(valid_offdiag, False)

    positive = distances[(valid_offdiag) & (distances > 0)]
    if positive.size == 0:
        return scores

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_distances = np.where(
        valid_offdiag,
        np.maximum(distances, np.finfo(float).eps),
        np.inf,
    )

    inverse_distance = np.zeros_like(distances, dtype=float)
    inverse_distance[valid_offdiag] = np.exp(
        -safe_distances[valid_offdiag] / scale
    )

    row_min = np.min(safe_distances, axis=1, keepdims=True)
    col_min = np.min(safe_distances, axis=0, keepdims=True)

    row_advantage = np.zeros_like(distances, dtype=float)
    col_advantage = np.zeros_like(distances, dtype=float)

    row_advantage[valid_offdiag] = np.exp(
        -(safe_distances[valid_offdiag] - np.broadcast_to(row_min, distances.shape)[valid_offdiag])
        / (scale + row_min.repeat(n, axis=1)[valid_offdiag])
    )
    col_advantage[valid_offdiag] = np.exp(
        -(safe_distances[valid_offdiag] - np.broadcast_to(col_min, distances.shape)[valid_offdiag])
        / (scale + col_min.repeat(n, axis=0)[valid_offdiag])
    )

    row_sum = np.sum(inverse_distance, axis=1, keepdims=True)
    col_sum = np.sum(inverse_distance, axis=0, keepdims=True)

    row_competition = np.divide(
        inverse_distance,
        row_sum,
        out=np.zeros_like(inverse_distance),
        where=row_sum > 0,
    )
    col_competition = np.divide(
        inverse_distance,
        col_sum,
        out=np.zeros_like(inverse_distance),
        where=col_sum > 0,
    )

    rarity = np.sqrt(row_competition * col_competition)
    local_preference = np.sqrt(row_advantage * col_advantage)

    scores = (
        0.45 * inverse_distance
        + 0.25 * local_preference
        + 0.20 * rarity
        + 0.10 * np.sqrt(row_competition + col_competition)
    )

    scores[~valid_offdiag] = 0.0

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores.astype(float, copy=False)
