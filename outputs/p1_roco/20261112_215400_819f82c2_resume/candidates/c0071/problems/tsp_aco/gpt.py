import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    d = np.maximum(distance_matrix, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    finite_edges = d[np.isfinite(d)]
    if finite_edges.size == 0:
        return np.zeros((n, n), dtype=float)

    global_median = float(np.median(finite_edges))
    global_q25, global_q75 = np.quantile(finite_edges, [0.25, 0.75])
    global_iqr = max(float(global_q75 - global_q25), eps)

    row_median = np.median(d, axis=1)
    col_median = np.median(d, axis=0)

    row_q25 = np.quantile(d, 0.25, axis=1)
    row_q75 = np.quantile(d, 0.75, axis=1)
    col_q25 = np.quantile(d, 0.25, axis=0)
    col_q75 = np.quantile(d, 0.75, axis=0)

    row_iqr = np.maximum(row_q75 - row_q25, eps)
    col_iqr = np.maximum(col_q75 - col_q25, eps)

    row_scale = (
        0.45 * row_iqr
        + 0.35 * global_iqr
        + 0.20 * np.maximum(row_median, eps)
    )
    col_scale = (
        0.45 * col_iqr
        + 0.35 * global_iqr
        + 0.20 * np.maximum(col_median, eps)
    )
    row_scale = np.maximum(row_scale, eps)
    col_scale = np.maximum(col_scale, eps)

    safe_d = np.where(np.isfinite(d), d, 0.0)

    row_relative = (safe_d - row_median[:, None]) / row_scale[:, None]
    col_relative = (safe_d - col_median[None, :]) / col_scale[None, :]

    row_shortness = 1.0 / (1.0 + np.exp(np.clip(row_relative, -40.0, 40.0)))
    col_shortness = 1.0 / (1.0 + np.exp(np.clip(col_relative, -40.0, 40.0)))

    valid = np.isfinite(d)

    row_less = np.sum(
        (safe_d[:, :, None] > safe_d[:, None, :]) & valid[:, None, :],
        axis=2,
    )
    row_equal = np.sum(
        (safe_d[:, :, None] == safe_d[:, None, :]) & valid[:, None, :],
        axis=2,
    )
    col_less = np.sum(
        (safe_d[:, None, :] > safe_d[:, :, None]) & valid[None, :, :],
        axis=2,
    )
    col_equal = np.sum(
        (safe_d[:, None, :] == safe_d[:, :, None]) & valid[None, :, :],
        axis=2,
    )

    row_midrank = row_less + 0.5 * np.maximum(row_equal - 1, 0)
    col_midrank = col_less + 0.5 * np.maximum(col_equal - 1, 0)

    rank_denominator = max(n - 1, 1)
    row_rank_quality = np.exp(-row_midrank / (0.42 * rank_denominator))
    col_rank_quality = np.exp(-col_midrank / (0.42 * rank_denominator))

    mutual_neighborhood = np.sqrt(row_rank_quality * col_rank_quality)

    row_sorted = np.sort(safe_d, axis=1)
    row_sorted[:, 0] = np.where(
        np.isfinite(row_sorted[:, 0]), row_sorted[:, 0], global_median
    )
    best_out = row_sorted[:, 0]
    second_out = row_sorted[:, min(1, n - 1)]

    continuation_scale = np.maximum(
        0.5 * global_iqr + 0.5 * np.median(row_scale),
        eps,
    )

    continuation = np.exp(
        -np.maximum(second_out[None, :] - safe_d, 0.0)
        / continuation_scale
    )
    continuation *= np.exp(
        -np.maximum(safe_d - best_out[None, :], 0.0)
        / (2.0 * continuation_scale)
    )

    global_z = (safe_d - global_median) / global_iqr
    global_competitiveness = 1.0 / (
        1.0 + np.exp(np.clip(global_z, -40.0, 40.0))
    )

    flat_edges = finite_edges
    if flat_edges.size >= 4:
        q10, q90 = np.quantile(flat_edges, [0.10, 0.90])
        band_width = max(float(q90 - q10) / 8.0, eps)
    else:
        band_width = global_iqr

    band_distance = np.abs(
        safe_d[:, :, None] - safe_d[None, None, :]
    )
    band_density = np.sum(
        np.exp(-band_distance / band_width)
        * valid[None, None, :],
        axis=2,
    )
    edge_count = max(int(finite_edges.size), 1)
    rarity = 1.0 / (1.0 + band_density / edge_count * 8.0)
    rarity = 0.65 + 0.35 * rarity

    directional_score = (
        0.30 * np.sqrt(row_shortness * col_shortness)
        + 0.22 * np.sqrt(row_rank_quality * col_rank_quality)
        + 0.18 * mutual_neighborhood
        + 0.17 * continuation
        + 0.13 * global_competitiveness
    )

    score = directional_score * rarity
    score = np.where(valid, score, 0.0)

    reverse_score = score.T
    score = 0.82 * score + 0.18 * np.sqrt(
        np.maximum(score, 0.0) * np.maximum(reverse_score, 0.0)
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > eps:
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
