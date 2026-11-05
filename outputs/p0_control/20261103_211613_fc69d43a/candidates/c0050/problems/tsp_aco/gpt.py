import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-score matrix for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D NumPy array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    if np.any(distances[finite] < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    valid = finite & ~np.eye(n, dtype=bool) & (distances > 0)
    positive = distances[valid]

    if positive.size == 0:
        return scores

    fallback = float(np.median(positive))
    eps = np.finfo(float).eps

    work = np.where(valid, distances, fallback)
    np.fill_diagonal(work, fallback)

    local_scales = np.empty(n, dtype=float)
    for i in range(n):
        candidates = distances[valid[i]]
        local_scales[i] = (
            float(np.median(candidates))
            if candidates.size
            else fallback
        )

    local_scales = np.maximum(local_scales, eps)
    endpoint_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )

    normalized = work / np.maximum(endpoint_scale, eps)

    direct_affinity = np.exp(-normalized) / (1.0 + normalized)

    row_median = np.median(work, axis=1)
    competitiveness = np.minimum(
        row_median[:, None] / np.maximum(work, eps),
        row_median[None, :] / np.maximum(work, eps),
    )
    competitiveness = np.clip(competitiveness, 0.0, 2.0)

    neighborhood_mass = direct_affinity.sum(axis=1)
    mass_scale = float(np.median(neighborhood_mass))
    if not np.isfinite(mass_scale) or mass_scale <= 0:
        mass_scale = 1.0

    scarcity = mass_scale / (neighborhood_mass + mass_scale)
    endpoint_bonus = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    two_hop = np.min(
        work[:, :, None] + work[None, :, :],
        axis=1,
    )
    bridge_quality = 1.0 / (
        1.0 + two_hop / np.maximum(work, eps)
    )

    raw = (
        direct_affinity
        * (0.7 + 0.3 * competitiveness)
        * (0.8 + 0.2 * endpoint_bonus)
        * (0.85 + 0.15 * bridge_quality)
    )

    raw = 0.5 * (raw + raw.T)
    raw[~valid] = 0.0
    np.fill_diagonal(raw, 0.0)

    values = raw[raw > 0]
    if values.size == 0:
        return scores

    minimum = float(values.min())
    maximum = float(values.max())

    if maximum > minimum:
        scores = np.where(
            raw > 0,
            (raw - minimum) / (maximum - minimum),
            0.0,
        )
    else:
        scores = np.where(raw > 0, 1.0, 0.0)

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    return scores.astype(float, copy=False)
