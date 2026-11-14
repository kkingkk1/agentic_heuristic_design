import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, direction-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=float)

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    valid = offdiag & np.isfinite(raw)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    observed = np.maximum(raw[valid], 0.0)
    positive = observed[observed > 0.0]

    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    penalty = maximum + 12.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = np.maximum(raw[valid], 0.0)
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps
    temperature = max(0.32 * scale, eps)

    scaled = np.minimum(distances / temperature, 60.0)
    affinity = np.exp(-scaled)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_max = np.maximum(np.max(affinity, axis=1, keepdims=True), eps)
    affinity = affinity / row_max

    incoming = affinity.sum(axis=0)
    incoming_max = max(float(np.max(incoming)), eps)
    destination_potential = incoming / incoming_max

    rank_order = np.argsort(
        np.where(valid, distances, np.inf),
        axis=1,
        kind="stable",
    )

    ranks = np.empty((n, n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    for row in range(n):
        ranks[row, rank_order[row]] = rank_values

    ranks[~valid] = float(n)
    rank_temperature = max(2.4, 0.20 * float(n))
    local_rank = np.exp(-ranks / rank_temperature)
    local_rank[~valid] = 0.0

    nearest = np.min(np.where(valid, distances, np.inf), axis=1)
    masked_second = np.where(
        valid & (distances > nearest[:, None] + eps),
        distances,
        np.inf,
    )
    second = np.min(masked_second, axis=1)

    fallback_second = nearest + scale
    second = np.where(np.isfinite(second), second, fallback_second)

    local_gap_scale = np.maximum(second - nearest, 0.20 * scale)
    competition_gap = np.zeros((n, n), dtype=float)
    competition_gap[valid] = np.clip(
        (second[:, None] - distances)[valid] / local_gap_scale[:, None][valid],
        -8.0,
        8.0,
    )

    competition = np.zeros((n, n), dtype=float)
    competition[valid] = 0.5 + 0.5 * (
        1.0 / (1.0 + np.exp(-competition_gap[valid]))
    )

    destination = 0.72 + 0.28 * destination_potential[None, :]

    core = (
        np.power(np.maximum(affinity, eps), 0.62)
        * np.power(np.maximum(local_rank, eps), 0.23)
        * np.power(np.maximum(competition, eps), 0.15)
        * destination
    )

    core[~valid] = 0.0

    exploration = np.zeros((n, n), dtype=float)
    exploration[valid] = 0.55 * affinity[valid] + 0.45 * local_rank[valid]

    floor = 0.045
    result = (1.0 - floor) * core + floor * exploration
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    row_max = np.maximum(np.max(result, axis=1, keepdims=True), eps)
    result = result / row_max

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
