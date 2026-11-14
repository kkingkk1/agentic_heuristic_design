import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally normalized and endpoint-balanced TSP edge priors."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    finite = np.isfinite(distances)
    valid_offdiag = finite & (~diagonal) & (distances >= 0.0)
    positive = distances[valid_offdiag & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    fallback = max(fallback, 1.0)

    d = np.where(
        valid_offdiag,
        distances,
        fallback,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.nan)

    positive_d = d[(~diagonal) & (d > 0.0)]
    global_median = (
        float(np.median(positive_d))
        if positive_d.size
        else fallback
    )
    global_median = max(global_median, eps)

    row_median = np.nanmedian(work, axis=1)
    row_q25 = np.nanpercentile(work, 25.0, axis=1)
    row_q75 = np.nanpercentile(work, 75.0, axis=1)

    row_median = np.maximum(
        np.nan_to_num(row_median, nan=global_median),
        eps,
    )
    row_q25 = np.maximum(
        np.nan_to_num(row_q25, nan=global_median),
        eps,
    )
    row_q75 = np.maximum(
        np.nan_to_num(row_q75, nan=global_median),
        row_q25 + eps,
    )

    log_distance = np.log1p(d / global_median)
    log_typical = np.log1p(row_median / global_median)

    endpoint_deviation = np.abs(
        log_distance - 0.5 * (
            log_typical[:, None] + log_typical[None, :]
        )
    )

    endpoint_scale = np.sqrt(
        row_median[:, None] * row_median[None, :]
    )
    relative_distance = d / (endpoint_scale + eps)

    global_affinity = np.exp(
        -np.clip((d / global_median) ** 0.72, 0.0, 40.0)
    )

    distribution_width = np.sqrt(
        (row_q75 - row_q25)[:, None]
        * (row_q75 - row_q25)[None, :]
    )
    distribution_width = np.maximum(
        distribution_width,
        global_median * 1e-6,
    )

    scale_consistency = np.exp(
        -np.clip(
            np.abs(
                np.log1p(row_median[:, None] / global_median)
                - np.log1p(row_median[None, :] / global_median)
            ),
            0.0,
            30.0,
        )
    )

    local_fit = np.exp(
        -np.clip(
            endpoint_deviation
            * (0.8 + 0.2 * relative_distance),
            0.0,
            30.0,
        )
    )

    spread_factor = 1.0 / (
        1.0
        + np.clip(
            np.abs(d - endpoint_scale) / distribution_width,
            0.0,
            30.0,
        )
    )

    score = (
        global_affinity ** 0.55
        * local_fit ** 1.25
        * spread_factor ** 0.35
        * scale_consistency ** 0.45
    )

    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    for _ in range(6):
        row_sum = np.sum(score, axis=1)
        row_sum = np.maximum(row_sum, eps)
        scale = 1.0 / np.sqrt(row_sum)
        score *= scale[:, None] * scale[None, :]
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

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
