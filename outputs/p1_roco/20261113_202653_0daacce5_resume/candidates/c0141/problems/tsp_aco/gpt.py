import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    offdiag = ~np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & np.isfinite(raw.T) & offdiag

    if not np.any(valid):
        return result

    distances = np.maximum(0.5 * (raw + raw.T), 0.0)
    distances[~valid] = np.nan

    observed = distances[valid]
    positive = observed[observed > 0.0]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(base_scale) or base_scale <= 0.0:
        base_scale = 1.0

    q25, q50, q75, q90 = np.nanpercentile(observed, [25, 50, 75, 90])
    spread = max(float(q75 - q25), 0.25 * base_scale, np.finfo(float).eps)
    upper = max(float(q90), float(q75) + 2.0 * spread, base_scale)

    clipped = np.minimum(np.where(valid, distances, upper), upper)
    positive_clipped = clipped[valid & (clipped > 0.0)]
    distance_scale = (
        float(np.median(positive_clipped))
        if positive_clipped.size
        else base_scale
    )
    distance_scale = max(distance_scale, np.finfo(float).eps)

    # Stable local preference: reciprocal distance with a bounded robust scale.
    reciprocal = 1.0 / (clipped + 0.08 * distance_scale)
    reciprocal[~valid] = 0.0
    reciprocal_valid = reciprocal[valid]
    reciprocal_scale = max(
        float(np.percentile(reciprocal_valid, 90)),
        np.finfo(float).eps,
    )
    reciprocal_score = np.clip(reciprocal / reciprocal_scale, 0.0, 1.0)

    # Bidirectional rank: an edge is preferred only when competitive at both ends.
    rank_score = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_valid = valid[i]
        count = int(np.sum(row_valid))
        if count == 0:
            continue

        row_values = np.where(row_valid, clipped[i], np.inf)
        order = np.argsort(row_values, kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)

        denom = max(count - 1, 1)
        rank_score[i] = np.clip(1.0 - ranks / denom, 0.0, 1.0)
        rank_score[i, ~row_valid] = 0.0

    bidirectional_rank = np.sqrt(rank_score * rank_score.T)
    bidirectional_rank[~valid] = 0.0

    # Small long-term bridge signal: retain edges whose removal is costly.
    finite_for_paths = np.where(valid, clipped, np.inf)
    bypass = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        via_k = finite_for_paths[:, [k]] + finite_for_paths[[k], :]
        bypass = np.minimum(bypass, via_k)

    bypass_gain = np.maximum(bypass - clipped, 0.0)
    gain_scale = max(spread, 0.5 * distance_scale, np.finfo(float).eps)
    bridge_score = 1.0 - np.exp(-bypass_gain / gain_scale)
    bridge_score[~valid] = 0.0
    bridge_score = np.clip(0.5 * (bridge_score + bridge_score.T), 0.0, 1.0)

    # Temperature-controlled exploration prevents collapse onto only nearest edges.
    temperature = max(0.55 * spread, 0.15 * distance_scale)
    soft_affinity = np.exp(
        -np.minimum(clipped, upper) / max(temperature, np.finfo(float).eps)
    )
    soft_affinity[~valid] = 0.0
    affinity_scale = max(
        float(np.percentile(soft_affinity[valid], 90)),
        np.finfo(float).eps,
    )
    exploration = np.clip(soft_affinity / affinity_scale, 0.0, 1.0)

    score = (
        0.46 * reciprocal_score
        + 0.34 * bidirectional_rank
        + 0.12 * bridge_score
        + 0.08 * exploration
    )
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0

    # Preserve a small nonzero floor on every valid edge for global diversity.
    valid_scores = score[valid]
    floor = max(
        0.025 * float(np.percentile(valid_scores, 75)),
        1e-6,
    )
    score[valid] = np.maximum(score[valid], floor)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
