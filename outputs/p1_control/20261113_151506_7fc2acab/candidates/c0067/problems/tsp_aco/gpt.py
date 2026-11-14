import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    if n == 1:
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    # Use a slightly broader robust neighborhood scale.
    sorted_rows = np.sort(work, axis=1)
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n) * 1.25))))
    local_scale = np.mean(sorted_rows[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    # Emphasize edges that are short relative to both endpoints.
    shortness = np.exp(
        -1.15 * work
        / (local_scale[:, None] + local_scale[None, :] + eps)
    )

    # Mutual rank preference with a gentler decay.
    ranks = np.empty_like(work)
    order = np.argsort(work, axis=1, kind="stable")
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n)[None, :]
    ranks = np.maximum(ranks, 1.0)

    rank_scale = max(1.0, 1.25 * np.sqrt(n))
    mutual_preference = np.exp(
        -0.5
        * (
            ranks / rank_scale
            + ranks.T / rank_scale
        )
    )

    # Penalize edges dominated by competitive two-hop alternatives.
    via_cost = work[:, :, None] + work[None, :, :]
    direct = work[:, None, :]
    detour_ratio = (via_cost - direct) / (via_cost + direct + eps)

    idx = np.arange(n)
    valid_intermediate = np.ones((n, n, n), dtype=bool)
    valid_intermediate[idx, :, idx] = False
    valid_intermediate[:, idx, idx] = False
    valid_intermediate &= np.isfinite(via_cost)

    detour_ratio = np.where(valid_intermediate, detour_ratio, np.nan)
    best_detour = np.nanmin(detour_ratio, axis=1)
    best_detour = np.nan_to_num(best_detour, nan=0.0)

    detour_resistance = 0.5 + 0.5 * np.tanh(2.5 * best_detour)

    score = (
        shortness
        * (0.50 + 0.50 * mutual_preference)
        * (0.68 + 0.32 * detour_resistance)
    )

    # Preserve stability for undirected TSP instances.
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
