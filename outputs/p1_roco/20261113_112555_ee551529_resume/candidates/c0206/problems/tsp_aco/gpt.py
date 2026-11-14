import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(finite_nonnegative):
        return result

    forward = np.where(finite_nonnegative, raw, np.inf)
    safe = np.minimum(forward, forward.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return result

    np.fill_diagonal(safe, np.inf)

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    usable = min(6, n - 1)
    ordered = np.sort(safe, axis=1)
    neighborhood = ordered[:, :usable]
    neighborhood_valid = np.isfinite(neighborhood)

    counts = neighborhood_valid.sum(axis=1)
    local_scale = np.sum(
        np.where(neighborhood_valid, neighborhood, 0.0),
        axis=1,
    ) / np.maximum(counts, 1)

    local_scale = np.where(
        counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    edge_distance = np.where(valid, safe, 0.0)

    # Soft distance attraction with a logarithmic compression for outliers.
    compressed = np.log1p(edge_distance / global_scale)
    distance_term = np.exp(
        -np.clip(compressed * global_scale / pair_scale, 0.0, 40.0)
    )

    # Preference for edges that are competitive with the best alternatives
    # at both endpoints.
    best = np.where(
        np.isfinite(ordered[:, 0]),
        ordered[:, 0],
        global_scale,
    )
    second = np.where(
        usable > 1,
        np.where(np.isfinite(ordered[:, 1]), ordered[:, 1], global_scale),
        best,
    )

    endpoint_scale = np.maximum(
        0.5 * (best + second),
        global_scale * 0.05,
    )

    row_opportunity = np.clip(
        (edge_distance - best[:, None]) / endpoint_scale[:, None],
        0.0,
        8.0,
    )
    col_opportunity = np.clip(
        (edge_distance - best[None, :]) / endpoint_scale[None, :],
        0.0,
        8.0,
    )

    opportunity_term = np.exp(
        -0.55 * np.sqrt(row_opportunity * col_opportunity)
    )

    # Estimate whether the edge has a short two-hop alternative through a
    # common intermediate node. Large detours indicate structural usefulness.
    two_hop = safe[:, :, None] + safe[None, :, :]
    common = (
        np.isfinite(safe[:, :, None])
        & np.isfinite(safe[None, :, :])
    )

    detour = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        candidate = safe[:, k][:, None] + safe[k, :][None, :]
        candidate_valid = (
            np.isfinite(safe[:, k])[:, None]
            & np.isfinite(safe[k, :])[None, :]
        )
        detour = np.minimum(
            detour,
            np.where(candidate_valid, candidate, np.inf),
        )

    detour_ratio = np.divide(
        detour,
        np.maximum(safe, eps),
        out=np.full_like(safe, np.inf),
        where=valid & np.isfinite(detour),
    )

    # Edges whose best two-hop replacement is much longer receive support.
    structural_term = 1.0 + 0.22 * np.tanh(
        np.clip(detour_ratio - 1.0, 0.0, 6.0)
    )

    # Mildly favor edges incident to nodes with fewer usable alternatives.
    degree = valid.sum(axis=1).astype(float)
    scarcity = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)
    scarcity_term = 0.88 + 0.24 * scarcity_pair

    scores = (
        distance_term
        * opportunity_term
        * structural_term
        * scarcity_term
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores = np.maximum(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
