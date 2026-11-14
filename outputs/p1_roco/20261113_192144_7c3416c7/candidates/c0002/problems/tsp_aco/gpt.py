import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    safe_d = np.where(positive, d, np.inf)

    inv_distance = np.zeros_like(safe_d, dtype=float)
    inv_distance[positive] = 1.0 / safe_d[positive]

    row_scale = np.max(inv_distance, axis=1, keepdims=True)
    row_scale[row_scale <= 0] = 1.0
    local_preference = inv_distance / row_scale

    col_scale = np.max(inv_distance, axis=0, keepdims=True)
    col_scale[col_scale <= 0] = 1.0
    local_preference = 0.5 * (
        local_preference + inv_distance / col_scale
    )

    valid_values = d[positive]
    if valid_values.size:
        distance_scale = np.median(valid_values)
        if not np.isfinite(distance_scale) or distance_scale <= 0:
            distance_scale = 1.0
    else:
        distance_scale = 1.0

    global_preference = 1.0 / (1.0 + np.maximum(d, 0.0) / distance_scale)
    global_preference[~finite] = 0.0

    score = 0.75 * local_preference + 0.25 * global_preference
    score = 0.5 * (score + score.T)
    score[~positive & ~positive.T] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
