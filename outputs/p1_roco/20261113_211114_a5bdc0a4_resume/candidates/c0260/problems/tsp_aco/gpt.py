import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, directional edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(raw)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.where(valid, np.maximum(raw, 0.0), 0.0)
    valid_values = distances[valid]

    positive = valid_values[valid_values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    def row_normalize(values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values, dtype=float)

        for i in range(n):
            mask = valid[i]
            if not np.any(mask):
                continue

            row = values[i, mask]
            low = float(np.quantile(row, 0.10))
            high = float(np.quantile(row, 0.90))
            width = max(high - low, 0.12 * scale, np.finfo(float).eps)

            result[i, mask] = np.clip((row - low) / width, 0.0, 1.0)

        return result

    affinity_raw = 1.0 / (1.0 + distances / scale)
    affinity = row_normalize(affinity_raw)

    rank_score = np.zeros((n, n), dtype=float)
    local_regret = np.zeros((n, n), dtype=float)
    rarity = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        ordered = indices[
            np.argsort(distances[i, indices], kind="stable")
        ]

        denominator = max(ordered.size - 1, 1)
        for rank, j in enumerate(ordered):
            rank_score[i, j] = 1.0 - rank / denominator

        best_distance = float(distances[i, ordered[0]])
        if ordered.size > 1:
            second_distance = float(distances[i, ordered[1]])
            gap_scale = max(
                second_distance - best_distance,
                0.12 * scale,
                np.finfo(float).eps,
            )
            local_regret[i, ordered[0]] = np.clip(
                (second_distance - best_distance) / gap_scale,
                0.0,
                1.0,
            )
        else:
            local_regret[i, ordered[0]] = 1.0

        near_threshold = float(np.quantile(distances[i, indices], 0.35))
        near_count = max(
            int(np.sum(distances[i, indices] <= near_threshold)),
            1,
        )
        node_rarity = 1.0 / np.sqrt(float(near_count))
        rarity[i, indices] = node_rarity

    regret_scale = max(
        float(np.quantile(local_regret[valid], 0.75))
        if np.any(local_regret[valid] > 0.0)
        else 1.0,
        np.finfo(float).eps,
    )
    local_regret = np.clip(local_regret / regret_scale, 0.0, 1.0)

    bypass = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        left_valid = valid[:, k][:, None]
        right_valid = valid[k, :][None, :]
        candidate = distances[:, k][:, None] + distances[k, :][None, :]
        candidate = np.where(left_valid & right_valid, candidate, np.inf)
        bypass = np.minimum(bypass, candidate)

    bypass_advantage = np.zeros((n, n), dtype=float)
    usable_bypass = valid & np.isfinite(bypass)
    bypass_advantage[usable_bypass] = np.maximum(
        bypass[usable_bypass] - distances[usable_bypass],
        0.0,
    )

    lookahead = row_normalize(bypass_advantage)

    rarity_values = rarity[valid]
    rarity_low = float(np.quantile(rarity_values, 0.10))
    rarity_high = float(np.quantile(rarity_values, 0.90))
    rarity_width = max(
        rarity_high - rarity_low,
        np.finfo(float).eps,
    )
    rarity_norm = np.zeros((n, n), dtype=float)
    rarity_norm[valid] = np.clip(
        (rarity_values - rarity_low) / rarity_width,
        0.0,
        1.0,
    )

    bridge_diversity = 0.5 * (
        rarity_norm + rarity_norm.T
    )
    bridge_diversity = np.where(valid, bridge_diversity, 0.0)

    score = (
        0.46 * affinity
        + 0.24 * rank_score
        + 0.16 * local_regret
        + 0.10 * lookahead
        + 0.04 * bridge_diversity
    )

    score[valid] = 0.96 * score[valid] + 0.04
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
