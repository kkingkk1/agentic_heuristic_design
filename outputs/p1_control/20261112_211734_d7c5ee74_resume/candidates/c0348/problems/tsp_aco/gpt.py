import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive mutual-neighborhood edge priorities for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, np.inf)

    eps = np.finfo(float).eps
    scale_floor = max(float(np.median(positive)) * 1e-10, eps)

    k = max(1, min(n - 1, int(np.sqrt(n))))

    nearest_values = np.partition(
        repaired,
        kth=k - 1,
        axis=1,
    )[:, :k]

    local_scale = np.maximum(
        np.mean(nearest_values, axis=1),
        scale_floor,
    )

    if n > 2:
        alternatives = np.partition(repaired, kth=1, axis=1)[:, 1]
    else:
        alternatives = nearest_values[:, 0]

    alternatives = np.where(
        np.isfinite(alternatives),
        alternatives,
        local_scale,
    )
    alternatives = np.maximum(alternatives, scale_floor)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = repaired / (pair_scale + scale_floor)

    distance_affinity = np.exp(
        -np.clip(normalized_distance, 0.0, 40.0)
    )

    endpoint_i = np.exp(
        -np.clip(
            repaired / (local_scale[:, None] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_j = np.exp(
        -np.clip(
            repaired / (local_scale[None, :] + scale_floor),
            0.0,
            40.0,
        )
    )
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    kth_cutoff = np.maximum(
        nearest_values[:, -1],
        scale_floor,
    )

    temperature_i = 0.25 * kth_cutoff + scale_floor
    temperature_j = 0.25 * kth_cutoff + scale_floor

    gate_i = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (repaired - kth_cutoff[:, None]) / temperature_i[:, None],
                -40.0,
                40.0,
            )
        )
    )
    gate_j = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (repaired - kth_cutoff[None, :]) / temperature_j[None, :],
                -40.0,
                40.0,
            )
        )
    )
    mutual_neighbor_gate = np.sqrt(gate_i * gate_j)

    alternative_scale = np.sqrt(
        alternatives[:, None] * alternatives[None, :]
    )
    relative_margin = (
        alternative_scale - repaired
    ) / (
        alternative_scale + repaired + scale_floor
    )
    relative_margin = np.clip(relative_margin, 0.0, 1.0)

    margin_bonus = 1.0 + 0.10 * relative_margin * mutual_neighbor_gate

    score = (
        distance_affinity
        * endpoint_compatibility
        * (0.35 + 0.65 * mutual_neighbor_gate)
        * margin_bonus
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
