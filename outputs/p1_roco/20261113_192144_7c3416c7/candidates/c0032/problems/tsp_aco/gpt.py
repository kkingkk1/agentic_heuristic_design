import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed TSP edge priors from soft graph diffusion and detours."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    usable = distances[finite]

    if usable.size == 0:
        return np.zeros_like(distances, dtype=float)

    usable = np.maximum(usable, 0.0)
    positive = usable[usable > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.max(usable))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_max = float(np.max(usable)) if usable.size else scale
    replacement = max(finite_max + 8.0 * scale, scale)

    clean = np.where(finite, np.maximum(distances, 0.0), replacement)
    clean = clean.astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    off_diagonal = ~np.eye(n, dtype=bool)
    positive_edges = clean[off_diagonal & (clean > 0.0)]

    if positive_edges.size:
        low = float(np.percentile(positive_edges, 25.0))
        high = float(np.percentile(positive_edges, 75.0))
        spread = max(high - low, 0.05 * scale, np.finfo(float).eps)
    else:
        spread = scale

    temperature = max(0.5 * spread, 0.05 * scale, np.finfo(float).eps)

    affinity = np.exp(-np.clip(clean / temperature, 0.0, 60.0))
    affinity[~off_diagonal] = 0.0
    affinity[~finite] = 0.0

    if not np.any(affinity > 0.0):
        return np.zeros_like(distances, dtype=float)

    # Approximate a directed doubly-stochastic kernel.
    kernel = affinity.copy()
    for _ in range(8):
        row_sum = kernel.sum(axis=1, keepdims=True)
        kernel /= np.maximum(row_sum, np.finfo(float).eps)

        col_sum = kernel.sum(axis=0, keepdims=True)
        kernel /= np.maximum(col_sum, np.finfo(float).eps)

    kernel = np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(kernel, 0.0)

    # Two-step alternatives for every directed edge.
    two_step_cost = clean @ np.ones((n, n), dtype=float)
    two_step_cost = np.full((n, n), np.inf, dtype=float)

    for middle in range(n):
        candidate = clean[:, middle, None] + clean[None, middle, :]
        two_step_cost = np.minimum(two_step_cost, candidate)

    np.fill_diagonal(two_step_cost, np.inf)

    direct = clean.copy()
    detour_margin = (two_step_cost - direct) / (temperature + scale * 1e-12)
    detour_margin = np.clip(detour_margin, -30.0, 30.0)

    # Large positive margin means the direct edge is structurally hard to replace.
    bridge_score = 1.0 / (1.0 + np.exp(-detour_margin))
    bridge_score[~np.isfinite(bridge_score)] = 0.0
    bridge_score[~off_diagonal] = 0.0
    bridge_score[~finite] = 0.0

    # Diffusion support captures multi-hop reachability without route sampling.
    two_hop = kernel @ kernel
    three_hop = two_hop @ kernel
    diffusion = 0.65 * two_hop + 0.35 * three_hop
    diffusion = np.maximum(diffusion, 0.0)

    diffusion_max = float(np.max(diffusion))
    if np.isfinite(diffusion_max) and diffusion_max > 0.0:
        diffusion /= diffusion_max

    direct_affinity = np.exp(-np.clip(direct / scale, 0.0, 60.0))
    direct_affinity[~off_diagonal] = 0.0
    direct_affinity[~finite] = 0.0

    # Penalize edges whose endpoints already have overwhelming alternative flow.
    outgoing = kernel.sum(axis=1, keepdims=True)
    incoming = kernel.sum(axis=0, keepdims=True)
    endpoint_balance = np.sqrt(
        np.maximum(kernel, 0.0)
        * np.maximum(kernel.T, 0.0)
    )
    endpoint_balance = 0.5 + 0.5 * endpoint_balance

    score = (
        0.35 * direct_affinity
        + 0.40 * bridge_score
        + 0.25 * diffusion
    )
    score *= endpoint_balance
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score.astype(float, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
