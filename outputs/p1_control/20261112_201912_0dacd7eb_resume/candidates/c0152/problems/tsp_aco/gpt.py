import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return triangle-support edge indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_values = d[off_diagonal & np.isfinite(d)]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    replacement = float(np.percentile(positive_values, 95.0))
    if not np.isfinite(replacement) or replacement <= 0.0:
        replacement = float(np.max(positive_values))
    if not np.isfinite(replacement) or replacement <= 0.0:
        replacement = 1.0

    d = np.where(np.isfinite(d), d, replacement)
    np.fill_diagonal(d, np.inf)

    valid_values = d[np.isfinite(d) & off_diagonal]
    if valid_values.size == 0:
        return np.zeros((n, n), dtype=float)

    q25, q50, q75 = np.percentile(valid_values, [25.0, 50.0, 75.0])
    iqr = max(float(q75 - q25), eps)
    global_median = max(float(q50), eps)

    finite_d = np.where(np.isfinite(d), d, replacement)

    row_values = finite_d.copy()
    np.fill_diagonal(row_values, np.nan)

    local_median = np.nanmedian(row_values, axis=1)
    local_median = np.maximum(local_median, eps)

    local_deviation = np.abs(
        row_values - local_median[:, None]
    )
    local_mad = np.nanmedian(local_deviation, axis=1)
    local_mad = np.maximum(1.4826 * local_mad, 0.05 * iqr)
    local_mad = np.maximum(local_mad, eps)

    robust_i = (
        (finite_d - local_median[:, None])
        / local_mad[:, None]
    )
    robust_j = (
        (finite_d - local_median[None, :])
        / local_mad[None, :]
    )

    endpoint_quality_i = 1.0 / (
        1.0 + np.maximum(robust_i, 0.0)
    )
    endpoint_quality_j = 1.0 / (
        1.0 + np.maximum(robust_j, 0.0)
    )

    endpoint_quality = np.minimum(
        endpoint_quality_i,
        endpoint_quality_j,
    )

    global_shortness = 1.0 / (
        1.0 + np.maximum(finite_d - q25, 0.0) / iqr
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_i = finite_d / local_median[:, None]
        normalized_j = finite_d / local_median[None, :]

    endpoint_balance = 1.0 / (
        1.0
        + np.abs(
            np.log1p(np.maximum(normalized_i, 0.0))
            - np.log1p(np.maximum(normalized_j, 0.0))
        )
    )

    if n > 2:
        third_mask = ~np.eye(n, dtype=bool)
        two_hop = (
            finite_d[:, :, None]
            + finite_d[None, :, :]
        )

        direct = finite_d[:, :, None]
        detour_slack = two_hop - direct

        detour_slack[:, :, :] = np.where(
            third_mask[None, :, :],
            detour_slack,
            0.0,
        )

        positive_slack = np.maximum(detour_slack, 0.0)

        slack_scale = (
            local_mad[:, None, None]
            + local_mad[None, :, None]
            + global_median
        )
        slack_strength = positive_slack / np.maximum(
            slack_scale,
            eps,
        )

        detour_support = np.sum(
            np.minimum(slack_strength, 4.0),
            axis=2,
        )

        possible_third = max(n - 2, 1)
        detour_support /= float(possible_third)

        direct_excess = np.maximum(
            finite_d - global_median,
            0.0,
        ) / iqr

        triangle_evidence = detour_support / (
            1.0 + 0.35 * direct_excess
        )
        triangle_evidence = triangle_evidence / (
            1.0 + triangle_evidence
        )
    else:
        triangle_evidence = np.zeros((n, n), dtype=float)

    score = (
        0.34 * endpoint_quality
        + 0.24 * global_shortness
        + 0.22 * triangle_evidence
        + 0.20 * endpoint_balance
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
