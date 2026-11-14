import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    clean = np.full((n, n), np.inf, dtype=float)

    both_valid = valid & valid.T
    clean[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])

    only_forward = valid & ~valid.T
    clean[only_forward] = d[only_forward]

    only_backward = ~valid & valid.T
    clean[only_backward] = d.T[only_backward]

    np.fill_diagonal(clean, np.inf)

    finite_values = clean[np.isfinite(clean)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0.0]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe_distance = np.where(np.isfinite(clean), clean, scale * 100.0)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(clean[valid], scale * 1e-12)

    row_max = inverse.max(axis=1, keepdims=True)
    normalized_inverse = inverse / np.maximum(row_max, np.finfo(float).eps)

    row_valid_count = valid.sum(axis=1)
    row_sum = normalized_inverse.sum(axis=1)
    local_scale = np.zeros(n, dtype=float)

    for i in range(n):
        if row_valid_count[i] > 0:
            local_scale[i] = float(np.mean(clean[i, valid[i]]))

    local_scale = np.maximum(local_scale, scale * 1e-6)

    relative_preference = np.exp(
        -safe_distance
        / np.sqrt(local_scale[:, None] * local_scale[None, :])
    )
    relative_preference[~valid] = 0.0

    local_strength = normalized_inverse / np.maximum(
        row_sum[:, None], np.finfo(float).eps
    )
    mutual_support = np.sqrt(local_strength * local_strength.T)
    mutual_support[~(valid | valid.T)] = 0.0

    scarcity = 1.0 / np.maximum(row_sum, np.finfo(float).eps)
    scarcity /= max(float(np.mean(scarcity)), np.finfo(float).eps)
    scarcity = np.clip(scarcity, 0.5, 3.0)
    endpoint_support = np.sqrt(scarcity[:, None] * scarcity[None, :])

    score = (
        0.50 * normalized_inverse
        + 0.22 * relative_preference
        + 0.18 * mutual_support
        + 0.10 * endpoint_support * np.sqrt(
            np.maximum(relative_preference, 0.0)
        )
    )

    score = 0.5 * (score + score.T)
    score[~(valid | valid.T)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score, initial=0.0))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
