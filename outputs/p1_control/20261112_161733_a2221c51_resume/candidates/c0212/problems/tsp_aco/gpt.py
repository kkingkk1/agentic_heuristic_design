import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps

    positive_values = d[valid & (d > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    safe_distance = np.where(valid, np.maximum(d, 0.0), 0.0)

    valid_counts = np.sum(valid, axis=1).astype(float)
    row_sum = np.sum(safe_distance, axis=1)
    row_mean = np.divide(
        row_sum,
        np.maximum(valid_counts, 1.0),
    )

    centered = np.where(
        valid,
        safe_distance - row_mean[:, None],
        0.0,
    )
    row_variance = np.divide(
        np.sum(centered * centered, axis=1),
        np.maximum(valid_counts, 1.0),
    )
    row_scale = np.sqrt(row_variance)
    row_scale = np.maximum(row_scale, 0.25 * global_scale)
    row_mean = np.maximum(row_mean, eps)

    standardized_edge = (
        safe_distance
        - 0.5 * (row_mean[:, None] + row_mean[None, :])
    ) / (
        row_scale[:, None]
        + row_scale[None, :]
        + eps
    )

    direct_affinity = 1.0 / (
        1.0 + np.exp(np.clip(standardized_edge, -40.0, 40.0))
    )

    node_scale = np.sqrt(row_mean * row_mean + row_scale * row_scale)
    node_scale = np.maximum(node_scale, eps)

    kernel_exponent = -(
        safe_distance
        / (node_scale[:, None] + eps)
    ) ** 2
    kernel = np.exp(np.clip(kernel_exponent, -40.0, 0.0))
    kernel[~valid] = 0.0
    np.fill_diagonal(kernel, 0.0)

    common_support = kernel @ kernel.T
    support_scale = np.sqrt(
        np.maximum(
            np.diag(common_support)[:, None]
            * np.diag(common_support)[None, :],
            eps,
        )
    )
    normalized_support = common_support / support_scale
    normalized_support = np.clip(normalized_support, 0.0, 1.0)

    scale_balance = 1.0 / (
        1.0
        + np.abs(
            np.log(
                (node_scale[:, None] + eps)
                / (node_scale[None, :] + eps)
            )
        )
    )

    score = (
        direct_affinity
        * (0.55 + 0.45 * normalized_support)
        * (0.65 + 0.35 * scale_balance)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
