import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric TSP edge-priority matrix."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)

    work = d.copy()
    np.fill_diagonal(work, np.nan)

    positive = work[valid & (work > 0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    row_median = np.nanmedian(work, axis=1)
    row_median = np.where(
        np.isfinite(row_median),
        row_median,
        global_scale,
    )

    absolute_deviation = np.abs(work - row_median[:, None])
    row_mad = np.nanmedian(absolute_deviation, axis=1)
    row_mad = np.where(
        np.isfinite(row_mad) & (row_mad > 0),
        row_mad,
        0.35 * np.maximum(row_median, global_scale),
    )
    row_mad = np.maximum(row_mad, np.finfo(float).eps)

    robust_z = (
        work - row_median[:, None]
    ) / (1.4826 * row_mad[:, None] + np.finfo(float).eps)

    endpoint_shortness = 1.0 / (
        1.0 + np.exp(np.clip(robust_z, -40.0, 40.0))
    )
    endpoint_shortness[~np.isfinite(endpoint_shortness)] = 0.0

    pair_shortness = np.sqrt(
        endpoint_shortness * endpoint_shortness.T
    )

    temperatures = np.maximum(
        0.5 * (row_mad + row_median),
        0.25 * global_scale,
    )
    soft_neighbor_mass = np.exp(
        -np.clip(work / temperatures[:, None], 0.0, 60.0)
    )
    soft_neighbor_mass[~valid] = 0.0

    total_mass = np.sum(soft_neighbor_mass, axis=1)
    edge_mass = soft_neighbor_mass
    alternative_mass_i = np.maximum(
        total_mass[:, None] - edge_mass,
        np.finfo(float).eps,
    )
    alternative_mass_j = np.maximum(
        total_mass[None, :] - edge_mass.T,
        np.finfo(float).eps,
    )

    scarcity_i = 1.0 / (
        1.0 + alternative_mass_i
    )
    scarcity_j = 1.0 / (
        1.0 + alternative_mass_j
    )
    mutual_scarcity = np.sqrt(scarcity_i * scarcity_j)

    global_shortness = np.exp(
        -np.clip(d / global_scale, 0.0, 60.0)
    )

    score = (
        pair_shortness
        * (0.55 + 1.10 * mutual_scarcity)
        * (0.70 + 0.30 * global_shortness)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
