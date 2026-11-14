import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structurally weighted edge priorities for stochastic TSP sampling."""
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

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = scale

    symmetric_valid = valid & valid.T

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    finite_work = np.where(np.isfinite(work), work, fallback)

    local_temperature = np.median(
        np.partition(finite_work, kth=max(0, min(n - 2, n // 2)), axis=1)
        [:, :max(1, min(n - 1, n // 2 + 1))],
        axis=1,
    )
    local_temperature = np.maximum(local_temperature, scale * 1e-10)

    temperature = np.sqrt(
        local_temperature[:, None] * local_temperature[None, :]
    )

    affinity = np.exp(
        -np.clip(d / (temperature + eps), 0.0, 60.0)
    )
    np.fill_diagonal(affinity, 0.0)

    logits = -finite_work / (
        local_temperature[:, None] + eps
    )
    logits = np.clip(logits, -60.0, 0.0)
    soft_weights = np.exp(logits)
    np.fill_diagonal(soft_weights, 0.0)

    row_mass = np.sum(soft_weights, axis=1, keepdims=True)
    probabilities = soft_weights / (row_mass + eps)

    mutual_preference = np.sqrt(
        probabilities * probabilities.T
    )

    endpoint_load = np.sum(
        np.sqrt(probabilities * probabilities.T),
        axis=1,
    )
    load_scale = np.mean(endpoint_load) + eps
    balance = 1.0 / (
        1.0
        + 0.35
        * np.maximum(
            endpoint_load[:, None] + endpoint_load[None, :]
            - 2.0 * load_scale,
            0.0,
        )
        / load_scale
    )

    two_hop = np.min(
        finite_work[:, :, None] + finite_work[None, :, :],
        axis=2,
    )
    np.fill_diagonal(two_hop, np.inf)

    detour_gap = (two_hop - d) / (
        two_hop + d + scale * 1e-10
    )
    detour_gap = np.nan_to_num(
        detour_gap,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    detour_gap = np.clip(detour_gap, -1.0, 1.0)

    bridge_support = 0.5 + 0.5 * np.tanh(
        2.5 * detour_gap
    )

    shared_support = np.sqrt(
        np.sum(
            probabilities[:, None, :] * probabilities[None, :, :],
            axis=2,
        )
    )

    score = (
        affinity
        * (0.25 + 0.75 * mutual_preference)
        * (0.35 + 0.65 * bridge_support)
        * (0.55 + 0.45 * shared_support)
        * balance
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
