import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise scores for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    valid = np.isfinite(d) & (d > 0.0)
    valid &= valid.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    valid_values = symmetric_d[valid]

    global_scale = max(float(np.median(valid_values)), eps)
    largest = max(float(np.max(valid_values)), global_scale)
    replacement = max(10.0 * largest, global_scale)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    # Use a scale based on several nearest neighbors instead of one edge.
    neighbor_count = min(max(1, int(np.sqrt(n))), n - 1)
    local_scale = np.partition(
        work, neighbor_count - 1, axis=1
    )[:, neighbor_count - 1]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = work / endpoint_scale
    global_length = work / global_scale

    # Continuous bilateral ranks.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)
    ranks[~valid] = 1.0

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - ranks, eps)
        * np.maximum(1.0 - ranks.T, eps)
    )

    # Similar local scales imply that the edge is geometrically compatible.
    log_scale_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.30 * np.minimum(log_scale_gap, 50.0)
    )

    # Soft endpoint competition: each endpoint distributes preference
    # among its incident edges, without hard candidate pruning.
    endpoint_logits = (
        -0.95 * (work / local_scale[:, None])
        -0.08 * global_length
    )
    endpoint_logits[~valid] = -np.inf

    row_max = np.max(endpoint_logits, axis=1, keepdims=True)
    shifted = endpoint_logits - row_max

    with np.errstate(over="ignore", invalid="ignore"):
        endpoint_prob = np.exp(np.clip(shifted, -745.0, 0.0))

    endpoint_prob[~valid] = 0.0
    row_sum = np.sum(endpoint_prob, axis=1, keepdims=True)
    endpoint_prob = endpoint_prob / np.maximum(row_sum, eps)

    bilateral_probability = np.sqrt(
        np.maximum(endpoint_prob, eps)
        * np.maximum(endpoint_prob.T, eps)
    )
    bilateral_probability[~valid] = eps

    # Penalize edges that are unusually long for either endpoint.
    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            0.70 * np.log(np.maximum(bilateral_probability, eps))
            + 0.55 * np.log(np.maximum(bilateral_rank, eps))
            + 0.35 * np.log(np.maximum(scale_compatibility, eps))
            - 0.35 * relative_length
            - 0.08 * global_length
            - 0.18 * excess_longness
        )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)

    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
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
