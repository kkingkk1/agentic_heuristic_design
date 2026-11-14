import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(raw)
    valid = finite & (raw >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    # Symmetrize only where at least one directed distance is usable.
    symmetric = np.zeros((n, n), dtype=float)
    both_valid = valid & valid.T
    symmetric[both_valid] = 0.5 * (raw[both_valid] + raw.T[both_valid])

    only_forward = valid & ~valid.T
    symmetric[only_forward] = raw[only_forward]
    symmetric[~np.isfinite(symmetric)] = np.inf
    np.fill_diagonal(symmetric, np.inf)

    positive = symmetric[np.isfinite(symmetric) & (symmetric > 0)]
    eps = np.finfo(float).eps
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, eps)

    # Zero or missing edges receive a finite fallback for scale estimation,
    # while invalid original entries remain suppressed in the final result.
    work = np.where(np.isfinite(symmetric), symmetric, fallback)
    work = np.where(work > 0, work, fallback)
    np.fill_diagonal(work, np.inf)

    k = min(max(2, int(np.sqrt(n))), n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = work / np.maximum(pair_scale, eps)

    proximity = np.exp(-np.clip(relative, 0.0, 60.0))

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_decay = max(1.0, np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_decay)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    best = nearest[:, 0]
    second = nearest[:, 1] if k >= 2 else nearest[:, 0]
    regret_gap = np.maximum(second - best, 0.0)
    node_regret = regret_gap / np.maximum(second, eps)
    node_regret = np.clip(node_regret, 0.0, 1.0)
    regret_bonus = 1.0 + 0.45 * (
        node_regret[:, None] + node_regret[None, :]
    )

    attractiveness = np.exp(
        -np.clip(work / np.maximum(local_scale[:, None], eps), 0.0, 60.0)
    )
    ambiguity = np.sum(attractiveness, axis=1)
    scarcity = 1.0 / np.sqrt(
        1.0 + ambiguity[:, None] + ambiguity[None, :]
    )

    score = (
        proximity
        * (0.55 + 0.45 * mutual_rank)
        * regret_bonus
        * (0.75 + 0.75 * scarcity)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
