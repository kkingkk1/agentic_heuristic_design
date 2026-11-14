import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe_d = np.where(np.isfinite(d), d, 0.0)
    symmetric_d = 0.5 * (safe_d + safe_d.T)
    symmetric_d = np.maximum(symmetric_d, 0.0)

    positive_values = symmetric_d[valid & (symmetric_d > eps)]
    if positive_values.size:
        global_scale = max(float(np.median(positive_values)), eps)
    else:
        global_scale = 1.0

    positive_sum = np.sum(
        np.where(valid & (symmetric_d > eps), symmetric_d, 0.0),
        axis=1,
    )
    positive_count = np.sum(
        valid & (symmetric_d > eps),
        axis=1,
    )

    row_mean = np.divide(
        positive_sum,
        np.maximum(positive_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=positive_count > 0,
    )

    temperature = np.maximum(
        0.35 * row_mean,
        0.08 * global_scale,
    )
    temperature = np.maximum(temperature, eps)

    row_min = np.min(
        np.where(valid, symmetric_d, np.inf),
        axis=1,
    )
    row_min = np.where(np.isfinite(row_min), row_min, 0.0)

    centered = symmetric_d - row_min[:, None]

    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.exp(
            -centered / temperature[:, None]
        )

    weights = np.where(valid & np.isfinite(weights), weights, 0.0)

    normalization = np.sum(weights, axis=1, keepdims=True)
    probabilities = np.divide(
        weights,
        np.maximum(normalization, eps),
        out=np.zeros_like(weights),
        where=normalization > 0.0,
    )

    # Bilateral soft-choice consistency.
    bilateral_choice = np.sqrt(
        np.maximum(probabilities * probabilities.T, 0.0)
    )

    # Measure whether an edge is substantially better than the alternatives
    # available to each endpoint.
    if n >= 2:
        second_choice = np.partition(
            probabilities,
            kth=n - 2,
            axis=1,
        )[:, n - 2]
    else:
        second_choice = np.zeros(n, dtype=float)

    endpoint_margin = probabilities / (
        probabilities + second_choice[:, None] + eps
    )
    bilateral_margin = np.sqrt(
        np.maximum(endpoint_margin * endpoint_margin.T, 0.0)
    )

    # Estimate whether a direct edge is a useful shortcut compared with
    # the best valid two-hop replacement.
    best_two_hop = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        via_k = (
            symmetric_d[:, k, None]
            + symmetric_d[None, k, :]
        )
        via_valid = (
            valid[:, k, None]
            & valid[None, k, :]
        )
        best_two_hop = np.minimum(
            best_two_hop,
            np.where(via_valid, via_k, np.inf),
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        shortcut_ratio = best_two_hop / np.maximum(symmetric_d, eps)

    shortcut_ratio = np.where(
        valid & np.isfinite(shortcut_ratio),
        shortcut_ratio,
        1.0,
    )
    shortcut_ratio = np.clip(shortcut_ratio, 0.25, 4.0)

    # Ratios above one indicate that the direct edge is shorter than its
    # best two-hop alternative.
    directness = np.power(shortcut_ratio, 0.65)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_score = (
            1.35 * np.log(np.maximum(bilateral_choice, eps))
            + 0.85 * np.log(np.maximum(bilateral_margin, eps))
            + 0.75 * np.log(np.maximum(directness, eps))
        )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(log_score[finite_scores]))

    with np.errstate(over="ignore", invalid="ignore"):
        score = np.exp(np.clip(log_score - shift, -745.0, 0.0))

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
