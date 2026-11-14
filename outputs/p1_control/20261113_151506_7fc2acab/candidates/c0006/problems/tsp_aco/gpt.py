import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    if not np.all(finite):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    if n == 1:
        return np.zeros_like(d, dtype=float)

    positive = work[np.isfinite(work) & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    # Robust endpoint-local distance scales.
    sorted_rows = np.sort(work, axis=1)
    k = max(1, min(n - 1, int(np.sqrt(n))))
    local_scale = np.mean(sorted_rows[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    # Shortness relative to both endpoints' neighborhoods.
    shortness = np.exp(
        -work / (local_scale[:, None] + local_scale[None, :] + eps)
    )

    # Mutual rank preference: edges favored by both endpoints receive more weight.
    ranks = np.empty_like(work)
    order = np.argsort(work, axis=1, kind="stable")
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n)[None, :]
    ranks = np.maximum(ranks, 1.0)
    rank_scale = max(1.0, np.sqrt(n))
    mutual_preference = np.exp(
        -0.5 * (ranks / rank_scale + ranks.T / rank_scale)
    )

    # Penalize edges that are strongly dominated by a two-hop detour.
    via_cost = work[:, :, None] + work[None, :, :]
    direct = work[:, None, :]
    detour_ratio = (via_cost - direct) / (via_cost + direct + eps)

    valid_intermediate = np.ones((n, n, n), dtype=bool)
    idx = np.arange(n)
    valid_intermediate[idx, :, idx] = False
    valid_intermediate[:, idx, idx] = False
    valid_intermediate &= np.isfinite(via_cost)

    detour_ratio = np.where(valid_intermediate, detour_ratio, np.nan)
    worst_detour = np.nanmin(detour_ratio, axis=1)
    worst_detour = np.nan_to_num(worst_detour, nan=0.0)
    detour_resistance = 0.5 + 0.5 * np.tanh(3.0 * worst_detour)

    # Mild symmetry improves stability for undirected TSP instances.
    score = (
        shortness
        * (0.55 + 0.45 * mutual_preference)
        * (0.65 + 0.35 * detour_resistance)
    )
    score = 0.5 * (score + score.T)

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
