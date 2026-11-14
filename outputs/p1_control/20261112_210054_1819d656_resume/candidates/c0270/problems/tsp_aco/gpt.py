import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric shortcut-oriented edge priorities for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    symmetric_valid = valid & valid.T
    positive = distances[valid & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    tau = max(0.35 * scale, eps)

    repaired = np.where(valid, distances, np.inf)
    repaired = np.where(
        np.isfinite(repaired) & np.isfinite(repaired.T),
        0.5 * (repaired + repaired.T),
        np.inf,
    )
    np.fill_diagonal(repaired, np.inf)

    direct = np.where(np.isfinite(repaired), repaired, 0.0)

    distance_affinity = np.exp(
        -np.clip(direct / (scale + eps), 0.0, 50.0)
    )

    if n > 2:
        via = repaired[:, :, None] + repaired[None, :, :]
        via = np.transpose(via, (0, 2, 1))

        indices = np.arange(n)
        intermediary_mask = (
            (indices[None, :, None] != indices[:, None, None])
            & (indices[None, :, None] != indices[None, None, :])
        )
        intermediary_mask = np.transpose(intermediary_mask, (0, 2, 1))

        via = np.where(intermediary_mask, via, np.inf)

        scaled = -via / tau
        row_max = np.max(scaled, axis=2)
        finite_max = np.isfinite(row_max)

        shifted = np.where(
            finite_max[:, :, None],
            scaled - row_max[:, :, None],
            -np.inf,
        )
        weights = np.exp(shifted)
        weight_sum = np.sum(weights, axis=2)

        soft_two_hop = np.full((n, n), np.inf, dtype=float)
        usable = finite_max & (weight_sum > 0.0)
        soft_two_hop[usable] = (
            -tau * (
                np.log(weight_sum[usable])
                + row_max[usable]
            )
        )

        relative_paths = np.zeros((n, n), dtype=float)
        relative_paths[usable] = np.clip(
            weight_sum[usable] / max(n - 2, 1),
            0.0,
            1.0,
        )

        shortcut_gap = np.zeros((n, n), dtype=float)
        shortcut_gap[usable] = np.clip(
            (soft_two_hop[usable] - direct[usable]) / tau,
            -20.0,
            20.0,
        )

        shortcut_advantage = np.zeros((n, n), dtype=float)
        shortcut_advantage[usable] = 1.0 / (
            1.0 + np.exp(-shortcut_gap[usable])
        )

        route_exclusivity = 1.0 - 0.18 * relative_paths
    else:
        shortcut_advantage = np.ones((n, n), dtype=float)
        route_exclusivity = np.ones((n, n), dtype=float)

    score = (
        distance_affinity
        * (0.55 + 0.45 * shortcut_advantage)
        * route_exclusivity
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
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
