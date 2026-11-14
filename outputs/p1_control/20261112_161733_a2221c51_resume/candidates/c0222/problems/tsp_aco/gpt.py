import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    positive_values = symmetric_d[valid]

    global_scale = max(float(np.median(positive_values)), eps)
    maximum_distance = max(float(np.max(positive_values)), global_scale)
    replacement = max(10.0 * maximum_distance, global_scale)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    # The second nearest valid neighbor gives a stable local scale.
    kth = min(1, n - 2)
    local_scale = np.partition(work, kth, axis=1)[:, kth]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = work / endpoint_scale
    global_length = work / global_scale

    # Continuous bilateral preference without explicit row-wise ranks.
    endpoint_preference = np.sqrt(
        local_scale[:, None] / np.maximum(work, eps)
        * local_scale[None, :] / np.maximum(work, eps)
    )
    endpoint_preference = np.maximum(endpoint_preference, eps)

    # Relative lengths act as a smooth reciprocal-rank surrogate.
    reciprocal_rank = 1.0 / (1.0 + relative_length)

    scale_mismatch = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(-0.32 * np.minimum(scale_mismatch, 50.0))

    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            0.95 * np.log(endpoint_preference)
            + 1.20 * np.log(np.maximum(reciprocal_rank, eps))
            + 0.45 * np.log(np.maximum(scale_compatibility, eps))
            - 0.38 * relative_length
            - 0.12 * global_length
            - 0.22 * excess_longness
        )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        score[finite_scores] = np.exp(
            np.clip(log_score[finite_scores] - shift, -745.0, 0.0)
        )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
