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

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)

    values = symmetric_d[valid]
    global_mean = max(float(np.mean(values)), eps)
    global_std = float(np.std(values))
    scale = max(global_std, 0.25 * global_mean, eps)

    largest = max(float(np.max(values)), global_mean)
    replacement = max(4.0 * largest, global_mean)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    finite_row_sum = np.sum(np.where(valid, symmetric_d, 0.0), axis=1)
    row_degree = np.sum(valid, axis=1)
    row_mean = np.divide(
        finite_row_sum,
        np.maximum(row_degree, 1),
        dtype=float,
    )
    row_mean = np.where(row_degree > 0, row_mean, global_mean)

    # Edges shorter than the typical incident edge receive positive utility.
    endpoint_saving = (
        (row_mean[:, None] - work)
        + (row_mean[None, :] - work)
    ) / (2.0 * scale)

    # Normalize edge cost globally without sorting or hard truncation.
    distance_utility = (global_mean - work) / scale

    # Estimate whether an edge is structurally important relative to
    # two-hop alternatives through every possible intermediate node.
    valid_i = valid.astype(float)
    two_hop = work[:, :, None] + work[None, :, :]
    two_hop_mask = valid_i[:, :, None] * valid_i[None, :, :]

    alternative_count = np.sum(two_hop_mask, axis=2)
    alternative_sum = np.sum(
        np.where(two_hop_mask > 0.0, two_hop, 0.0),
        axis=2,
    )

    mean_alternative = np.divide(
        alternative_sum,
        np.maximum(alternative_count, 1.0),
        dtype=float,
    )

    # Positive values indicate that replacing the direct edge by an
    # average two-hop route is relatively expensive.
    detour_gain = (mean_alternative - work) / scale
    detour_gain = np.tanh(detour_gain)

    # Penalize edges that are simultaneously expensive for both endpoints,
    # while retaining a continuous score for every valid edge.
    endpoint_pressure = (
        (work - row_mean[:, None]) / scale
        + (work - row_mean[None, :]) / scale
    ) / 2.0

    raw_score = (
        1.20 * distance_utility
        + 0.85 * endpoint_saving
        + 0.55 * detour_gain
        - 0.25 * np.tanh(endpoint_pressure)
    )

    raw_score = 0.5 * (raw_score + raw_score.T)
    raw_score[~valid] = -np.inf

    finite_scores = np.isfinite(raw_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    center = float(np.mean(raw_score[finite_scores]))
    spread = max(float(np.std(raw_score[finite_scores])), eps)

    with np.errstate(over="ignore", invalid="ignore"):
        score = 1.0 / (
            1.0 + np.exp(
                -np.clip((raw_score - center) / spread, -60.0, 60.0)
            )
        )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
