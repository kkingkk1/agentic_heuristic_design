import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust continuous edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances)
    valid = finite & off_diagonal & (distances >= 0.0)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_offdiag = d[np.isfinite(d)]
    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    global_median = float(np.median(finite_offdiag))
    global_q75 = float(np.percentile(finite_offdiag, 75.0))
    global_scale = max(global_q75 - global_median, eps)

    ordered = np.sort(d, axis=1)
    local_median = ordered[:, (n - 2) // 2]

    q25_index = max(0, int(np.floor(0.25 * (n - 1))) - 1)
    q75_index = min(n - 2, int(np.ceil(0.75 * (n - 1))) - 1)

    local_q25 = ordered[:, q25_index]
    local_q75 = ordered[:, q75_index]
    local_scale = np.maximum(local_q75 - local_q25, eps)

    local_median = np.maximum(local_median, eps)
    local_density = 1.0 / local_median

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    centered_distance = (
        d - 0.5 * (
            local_median[:, None] + local_median[None, :]
        )
    )

    centered_scale = np.maximum(pair_scale, eps)
    short_edge_probability = 1.0 / (
        1.0 + np.exp(
            np.clip(centered_distance / centered_scale, -40.0, 40.0)
        )
    )

    global_affinity = np.exp(
        -np.clip(
            (d / max(global_median, eps)) ** 0.85,
            0.0,
            40.0,
        )
    )

    endpoint_scale_i = np.maximum(local_scale[:, None], eps)
    endpoint_scale_j = np.maximum(local_scale[None, :], eps)

    selectivity_i = 1.0 / (
        1.0 + np.exp(
            np.clip(
                (d - local_median[:, None])
                / endpoint_scale_i,
                -40.0,
                40.0,
            )
        )
    )
    selectivity_j = 1.0 / (
        1.0 + np.exp(
            np.clip(
                (d - local_median[None, :])
                / endpoint_scale_j,
                -40.0,
                40.0,
            )
        )
    )

    joint_selectivity = np.sqrt(
        np.maximum(selectivity_i * selectivity_j, 0.0)
    )

    density_ratio = np.sqrt(
        np.maximum(
            local_density[:, None]
            / np.maximum(local_density[None, :], eps),
            eps,
        )
    )
    density_balance = np.exp(
        -0.30 * np.abs(np.log(np.maximum(density_ratio, eps)))
    )

    contrast = np.sqrt(
        np.maximum(short_edge_probability * joint_selectivity, 0.0)
    )

    score = (
        contrast
        * global_affinity
        * density_balance
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
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
