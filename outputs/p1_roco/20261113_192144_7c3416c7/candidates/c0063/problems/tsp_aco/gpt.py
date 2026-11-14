import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_positive = d[np.isfinite(d) & (d > 0.0)]
    if finite_positive.size == 0:
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    scale = max(
        float(np.median(finite_positive)),
        np.finfo(float).eps,
    )

    fallback_distance = scale * 80.0
    finite_d = np.where(np.isfinite(d), d, fallback_distance)

    row_min = np.min(finite_d, axis=1, keepdims=True)
    centered = np.maximum(finite_d - row_min, 0.0)

    lower_distances = np.sort(finite_positive)
    lower_reference = float(
        np.median(lower_distances[: max(1, min(lower_distances.size, 2 * n))])
    )
    temperature = max(
        0.28 * scale + 0.12 * lower_reference,
        np.finfo(float).eps,
    )

    short_edge_score = np.exp(-centered / temperature)

    nearest = np.argmin(finite_d, axis=1)
    mutual_bonus = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = int(nearest[i])
        if j != i and nearest[j] == i:
            mutual_bonus[i, j] = 1.0
            mutual_bonus[j, i] = 1.0

    sorted_d = np.sort(finite_d, axis=1)
    neighborhood_width = max(1, min(n - 1, 4))
    local_scale = np.mean(
        sorted_d[:, :neighborhood_width],
        axis=1,
    )
    local_scale = np.maximum(local_scale, scale * 0.04)

    denominator = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    sparsity_bonus = 1.0 / (1.0 + finite_d / denominator)

    score = (
        0.70 * short_edge_score
        + 0.20 * sparsity_bonus
        + 0.10 * mutual_bonus
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
