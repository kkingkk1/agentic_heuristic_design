import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, locally adaptive edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(raw) & (raw >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive = raw[valid & (raw > 0.0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = float(np.median(raw[valid]))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    # Construct a finite symmetric working distance view.
    safe = np.full((n, n), np.inf, dtype=float)
    both = valid & valid.T
    only_forward = valid & ~valid.T
    only_backward = ~valid & valid.T

    safe[both] = 0.5 * (raw[both] + raw.T[both])
    safe[only_forward] = raw[only_forward]
    safe[only_backward] = raw.T[only_backward]

    edge_valid = both | only_forward | only_backward
    np.fill_diagonal(safe, np.inf)
    np.fill_diagonal(edge_valid, False)

    if not np.any(edge_valid):
        return result

    valid_counts = edge_valid.sum(axis=1)
    work = np.where(edge_valid, safe, np.inf)

    # Robust endpoint-local scale from up to four nearest usable edges.
    local_scale = np.empty(n, dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = np.flatnonzero(edge_valid[i])
        if candidates.size == 0:
            local_scale[i] = global_scale
            continue

        distances = safe[i, candidates]
        order = np.argsort(distances, kind="stable")
        ordered_candidates = candidates[order]

        nearest[i] = int(ordered_candidates[0])
        k = min(4, distances.size)
        local_scale[i] = float(np.median(distances[order[:k]]))

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_shortness = np.exp(
            -np.clip(safe / pair_scale, 0.0, 50.0)
        )

    # Reciprocal endpoint rank consensus.
    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(edge_valid[i])
        if candidates.size == 0:
            continue

        order = np.argsort(safe[i, candidates], kind="stable")
        ranked = candidates[order]
        ranks[i, ranked] = 1.0 / (1.0 + np.arange(ranked.size))

    rank_consensus = np.sqrt(ranks * ranks.T)

    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Favor edges incident to nodes with fewer viable alternatives.
    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts.astype(float), 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    score = (
        relative_shortness
        * (0.55 + 0.45 * rank_consensus)
        * (1.0 + 0.30 * mutual)
        * (0.90 + 0.20 * scarcity_pair)
    )

    score[~edge_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
