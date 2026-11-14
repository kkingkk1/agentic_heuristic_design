import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structure-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = offdiag & (finite | finite.T)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    left_only = finite & ~finite.T
    right_only = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[left_only] = raw[left_only]
    cleaned[right_only] = raw.T[right_only]
    cleaned = np.maximum(cleaned, 0.0)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    penalty = maximum + 12.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = cleaned[valid]
    np.fill_diagonal(distances, penalty)
    distances = 0.5 * (distances + distances.T)

    eps = np.finfo(float).eps

    temperature = max(0.30 * scale, eps)
    affinity = np.exp(-np.minimum(distances / temperature, 60.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)
    affinity /= max(float(np.max(affinity)), eps)

    nearest = np.min(distances, axis=1)
    second = np.partition(distances, 1, axis=1)[:, 1]
    local_scale = np.maximum(second - nearest, 0.20 * scale)
    local_scale = np.maximum(local_scale, eps)

    competition = np.exp(
        np.clip((second[:, None] - distances) / local_scale[:, None], -20.0, 20.0)
    )
    competition[~valid] = 0.0
    competition = np.sqrt(competition * competition.T)

    candidate_count = min(12, n - 1)
    nearest_indices = np.argpartition(
        distances, kth=candidate_count - 1, axis=1
    )[:, :candidate_count]

    intermediates = np.unique(nearest_indices.ravel())
    bypass = np.full((n, n), penalty, dtype=float)

    for middle in intermediates:
        usable = valid[:, middle][:, None] & valid[middle, :][None, :]
        two_hop = (
            distances[:, middle][:, None] + distances[middle, :][None, :]
        )
        bypass = np.minimum(bypass, np.where(usable, two_hop, penalty))

    np.fill_diagonal(bypass, penalty)
    advantage = np.maximum(bypass - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -advantage / max(0.85 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    mst_score = np.zeros((n, n), dtype=float)

    in_tree = np.zeros(n, dtype=bool)
    best_cost = np.full(n, penalty, dtype=float)
    parent = np.full(n, -1, dtype=int)

    root = 0
    best_cost[root] = 0.0

    for _ in range(n):
        selectable = np.where(~in_tree, best_cost, np.inf)
        node = int(np.argmin(selectable))
        if not np.isfinite(selectable[node]):
            break

        in_tree[node] = True
        previous = parent[node]
        if previous >= 0 and valid[node, previous]:
            mst_score[node, previous] = 1.0
            mst_score[previous, node] = 1.0

        update = (~in_tree) & (distances[node] < best_cost)
        best_cost[update] = distances[node, update]
        parent[update] = node

    score = (
        0.46 * affinity
        + 0.25 * competition
        + 0.19 * irreplaceability
        + 0.10 * mst_score
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(10):
        row_sum = np.maximum(balanced.sum(axis=1, keepdims=True), eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.68 * score + 0.32 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
