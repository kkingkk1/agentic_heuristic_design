import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D NumPy array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)

    finite_off_diagonal = np.isfinite(distance_matrix)
    np.fill_diagonal(finite_off_diagonal, False)

    positive_distances = distance_matrix[
        finite_off_diagonal & (distance_matrix > 0)
    ]

    if positive_distances.size == 0:
        return np.zeros_like(distance_matrix, dtype=float)

    scale = float(np.median(positive_distances))
    scale = max(scale, np.finfo(float).eps)

    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix > 0)
    )
    np.fill_diagonal(valid, False)

    normalized_distance = np.divide(
        distance_matrix,
        scale,
        out=np.full_like(distance_matrix, np.inf, dtype=float),
        where=valid,
    )

    inverse_distance = np.divide(
        1.0,
        1.0 + normalized_distance,
        out=np.zeros_like(distance_matrix, dtype=float),
        where=valid,
    )

    local_distances = np.where(valid, distance_matrix, np.inf)

    if n > 1:
        nearest_distance = np.min(local_distances, axis=1)
        finite_nearest = np.isfinite(nearest_distance)

        fallback = float(np.median(positive_distances))
        nearest_distance = np.where(
            finite_nearest,
            nearest_distance,
            fallback,
        )

        scarcity = np.divide(
            nearest_distance,
            scale,
            out=np.ones(n, dtype=float),
            where=np.isfinite(nearest_distance),
        )
        scarcity = 1.0 + scarcity
    else:
        scarcity = np.ones(1, dtype=float)

    endpoint_weight = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    scores = inverse_distance * endpoint_weight
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
