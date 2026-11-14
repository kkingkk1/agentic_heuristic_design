import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric detour-advantage prior for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    safe = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    base_scale = max(base_scale, np.finfo(float).eps)

    finite_values = safe[np.isfinite(safe)]
    if finite_values.size:
        cost_scale = float(np.median(finite_values))
    else:
        return np.zeros((n, n), dtype=float)
    cost_scale = max(cost_scale, np.finfo(float).eps)

    # Robust endpoint scales describe the cost range near each node.
    ordered = np.sort(safe, axis=1)
    k = min(3, n - 1)
    neighborhood = ordered[:, :k]
    neighborhood = np.where(np.isfinite(neighborhood), neighborhood, np.nan)

    local_scale = np.nanmedian(neighborhood, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        cost_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    # Two-hop alternatives i -> k -> j, excluding invalid paths.
    two_hop = safe[:, :, None] + safe[None, :, :]
    # Shape above is (i, k, j) after transposition below.
    two_hop = safe[:, :, None] + safe[None, :, :]
    two_hop = np.transpose(two_hop, (0, 2, 1))
    # two_hop[i, j, k] = d[i, k] + d[k, j].
    np.fill_diagonal(two_hop[:, :, 0], np.inf) if n else None

    valid_paths = np.isfinite(two_hop)
    path_temperature = 0.35 * pair_scale[:, :, None] + np.finfo(float).eps

    path_affinity = np.exp(
        -np.minimum(two_hop / path_temperature, 50.0)
    )
    path_affinity = np.where(valid_paths, path_affinity, 0.0)

    path_mass = path_affinity.sum(axis=2)
    soft_detour = np.divide(
        (two_hop * path_affinity).sum(axis=2),
        path_mass,
        out=np.full((n, n), np.inf, dtype=float),
        where=path_mass > 0.0,
    )

    # Direct edges gain support when they beat their best plausible detour.
    detour_margin = (soft_detour - d) / pair_scale
    detour_signal = 1.0 / (1.0 + np.exp(-np.clip(detour_margin, -30.0, 30.0)))
    detour_signal = np.where(valid & np.isfinite(soft_detour), detour_signal, 0.0)

    # Estimate how consistently an edge is supported across intermediate nodes.
    competitive_paths = np.exp(
        -np.maximum(two_hop - d[:, :, None], 0.0)
        / (pair_scale[:, :, None] + np.finfo(float).eps)
    )
    competitive_paths = np.where(valid_paths, competitive_paths, 0.0)

    support_count = (competitive_paths > 0.35).sum(axis=2).astype(float)
    support_strength = competitive_paths.sum(axis=2)
    support = np.divide(
        support_strength,
        1.0 + support_count,
        out=np.zeros_like(support_strength),
        where=np.isfinite(support_strength),
    )
    support = 1.0 - np.exp(-support)
    support = np.where(valid, support, 0.0)

    # A bounded absolute-cost term prevents detour comparisons from
    # overvaluing edges in uniformly expensive regions.
    absolute_shortness = np.exp(
        -np.minimum(np.maximum(d, 0.0) / cost_scale, 40.0)
    )
    absolute_shortness = np.where(valid, absolute_shortness, 0.0)

    score = (
        0.62 * detour_signal
        + 0.23 * support
        + 0.15 * absolute_shortness
    )

    # Penalize only edges that are simultaneously poor against detours
    # and unusually long relative to both endpoints.
    relative_length = d / pair_scale
    penalty = np.where(relative_length > 3.0, 0.72, 1.0)
    score *= penalty

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
