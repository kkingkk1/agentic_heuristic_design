import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric positive edge-score matrix for a fixed TSP solver.

    Larger scores indicate edges that are preferred by the heuristic.
    Self-loops, invalid entries, and infinite edges receive zero score.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D NumPy array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    if np.any(distances[finite] < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    valid = finite & ~np.eye(n, dtype=bool)
    positive = valid & (distances > 0)

    positive_values = distances[positive]
    if positive_values.size == 0:
        return scores

    global_scale = float(np.median(positive_values))
    eps = max(global_scale * 1e-12, np.finfo(float).eps)

    # Use a symmetric working matrix for the undirected TSP setting.
    work = 0.5 * (distances + distances.T)
    work[~np.isfinite(work)] = np.inf
    np.fill_diagonal(work, np.inf)

    valid_work = np.isfinite(work)
    positive_work = valid_work & (work > 0)
    positive_values = work[positive_work]

    if positive_values.size == 0:
        return scores

    global_scale = float(np.median(positive_values))
    eps = max(global_scale * 1e-12, np.finfo(float).eps)

    # Adaptive local scale from the nearest usable neighbors.
    k = max(1, min(4, n - 1))
    local_scales = np.full(n, global_scale, dtype=float)

    for i in range(n):
        candidates = work[i, np.isfinite(work[i]) & (work[i] > 0)]
        if candidates.size:
            nearest = np.partition(candidates, min(k, candidates.size) - 1)
            local_scales[i] = float(
                np.mean(nearest[: min(k, candidates.size)])
            )

    local_scales = np.maximum(local_scales, eps)
    pair_scale = np.sqrt(local_scales[:, None] * local_scales[None, :])
    normalized = work / np.maximum(pair_scale, eps)

    usable = np.isfinite(work) & (work > 0)

    # Direct affinity favors edges short relative to both endpoints' neighborhoods.
    direct = np.zeros_like(work, dtype=float)
    direct[usable] = (
        np.exp(-normalized[usable])
        / (1.0 + normalized[usable])
    )

    # Endpoint scarcity emphasizes edges important to locally sparse cities.
    neighborhood_mass = direct.sum(axis=1)
    positive_mass = neighborhood_mass[neighborhood_mass > 0]
    mass_scale = (
        float(np.median(positive_mass))
        if positive_mass.size
        else 1.0
    )
    mass_scale = max(mass_scale, eps)

    scarcity = mass_scale / (neighborhood_mass + mass_scale)
    endpoint_bonus = np.sqrt(scarcity[:, None] * scarcity[None, :])

    # Bridge quality rewards edges that are competitive with two-hop alternatives.
    finite_work = np.where(np.isfinite(work), work, global_scale)
    two_hop = np.min(
        finite_work[:, :, None] + finite_work[None, :, :],
        axis=1,
    )
    bridge = np.zeros_like(work, dtype=float)
    bridge[usable] = np.maximum(
        (two_hop[usable] - work[usable])
        / (two_hop[usable] + work[usable] + eps),
        0.0,
    )

    score = direct * (0.55 + 0.45 * endpoint_bonus)
    score *= 1.0 + 0.25 * bridge

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
