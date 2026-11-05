import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return scores

    raw = distance_matrix.copy()
    finite = np.isfinite(raw)
    np.fill_diagonal(finite, False)

    if not np.any(finite):
        return scores

    distances = np.maximum(raw, 0.0)
    positive = distances[finite & (distances > 0.0)]

    if positive.size:
        global_scale = float(np.median(positive))
        scale_floor = max(
            global_scale * 1e-12,
            np.finfo(float).eps,
        )
    else:
        global_scale = 1.0
        scale_floor = np.finfo(float).eps

    row_scale = np.full(n, global_scale, dtype=float)
    k = min(3, n - 1)

    for i in range(n):
        row_values = distances[i, finite[i]]
        row_positive = row_values[row_values > 0.0]

        if row_positive.size:
            nearest = np.partition(row_positive, min(k, row_positive.size) - 1)
            row_scale[i] = float(np.median(nearest[:k]))

    row_scale = np.maximum(row_scale, scale_floor)

    safe_distances = np.where(
        finite,
        distances,
        max(float(np.max(distances[finite])), global_scale) + global_scale,
    )

    forward_ratio = safe_distances / row_scale[:, None]
    reverse_ratio = safe_distances / row_scale[None, :]

    log_affinity = (
        -np.log1p(forward_ratio)
        -np.log1p(reverse_ratio)
    )

    log_scale = np.log(row_scale)
    scale_gap = np.abs(log_scale[:, None] - log_scale[None, :])
    compatibility = np.exp(-scale_gap)

    log_score = log_affinity + np.log(0.35 + 0.65 * compatibility)
    log_score[~finite] = -np.inf
    np.fill_diagonal(log_score, -np.inf)

    valid_log_score = log_score[np.isfinite(log_score)]
    if valid_log_score.size == 0:
        return scores

    maximum = float(np.max(valid_log_score))
    scores = np.zeros_like(log_score, dtype=float)
    valid = np.isfinite(log_score)
    scores[valid] = np.exp(log_score[valid] - maximum)

    same_mask = finite == finite.T
    symmetric_input = bool(
        np.all(same_mask)
        and np.allclose(
            np.where(finite, distances, 0.0),
            np.where(finite.T, distances.T, 0.0),
            rtol=1e-10,
            atol=1e-12,
        )
    )

    if symmetric_input:
        scores = 0.75 * scores + 0.25 * scores.T

    np.fill_diagonal(scores, 0.0)
    scores[~finite] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return scores
