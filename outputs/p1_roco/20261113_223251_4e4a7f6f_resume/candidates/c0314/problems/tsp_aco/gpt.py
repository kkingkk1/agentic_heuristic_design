import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return sparse-aware directional edge priors for stochastic TSP sampling."""
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

    quantile = float(np.quantile(observed, 0.95))
    if not np.isfinite(quantile):
        quantile = scale

    distance_cap = max(quantile, scale)
    cleaned = np.full((n, n), distance_cap, dtype=float)
    cleaned[valid] = np.minimum(np.maximum(raw[valid], 0.0), distance_cap)
    np.fill_diagonal(cleaned, distance_cap)

    eps = np.finfo(float).eps

    reciprocal = 1.0 / (1.0 + cleaned / max(scale, eps))
    reciprocal[~valid] = 0.0

    row_order = np.argsort(cleaned, axis=1, kind="stable")
    row_rank = np.argsort(row_order, axis=1, kind="stable").astype(float)
    row_rank = np.exp(-row_rank / max(1.5, np.sqrt(float(n))))
    row_rank[~valid] = 0.0

    column_order = np.argsort(cleaned.T, axis=1, kind="stable")
    column_rank = np.argsort(column_order, axis=1, kind="stable").astype(float)
    column_rank = np.exp(-column_rank / max(1.5, np.sqrt(float(n))))
    column_rank[~valid] = 0.0

    nearest = np.min(cleaned, axis=1)
    second = np.partition(cleaned, 1, axis=1)[:, 1]
    local_gap = np.maximum(second - nearest, 0.20 * scale)

    local_competition = np.exp(
        -np.maximum(cleaned - nearest[:, None], 0.0)
        / np.maximum(local_gap[:, None], eps)
    )
    local_competition[~valid] = 0.0

    bridge = np.zeros((n, n), dtype=float)
    neighborhood = min(max(4, int(np.sqrt(float(n)) * 2.0)), 10, n - 1)

    for source in range(n):
        candidates = row_order[source]
        candidates = candidates[valid[source, candidates]][:neighborhood]

        if candidates.size == 0:
            continue

        best_two_step = np.full(n, distance_cap, dtype=float)

        for middle in candidates:
            if not np.isfinite(cleaned[source, middle]):
                continue

            reachable = valid[middle]
            candidate_cost = cleaned[source, middle] + cleaned[middle]
            best_two_step[reachable] = np.minimum(
                best_two_step[reachable],
                candidate_cost[reachable],
            )

        advantage = np.maximum(best_two_step - cleaned[source], 0.0)
        advantage = np.clip(advantage / max(0.80 * scale, eps), 0.0, 4.0)
        bridge[source] = 1.0 - np.exp(-advantage)

    bridge[~valid] = 0.0

    directional_balance = np.sqrt(
        np.maximum(row_rank, 0.0) * np.maximum(column_rank, 0.0)
    )
    directional_balance[~valid] = 0.0

    score = (
        0.44 * reciprocal
        + 0.24 * local_competition
        + 0.16 * directional_balance
        + 0.10 * bridge
        + 0.06 * np.sqrt(np.maximum(row_rank * column_rank, 0.0))
    )
    score[~valid] = 0.0

    positive_score = score[valid]
    floor = (
        0.025 * float(np.median(positive_score))
        if positive_score.size
        else 0.0
    )
    if floor > 0.0:
        score[valid] += floor

    row_mass = np.maximum(score.sum(axis=1, keepdims=True), eps)
    balanced = score / row_mass

    result = 0.78 * score + 0.22 * balanced
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
