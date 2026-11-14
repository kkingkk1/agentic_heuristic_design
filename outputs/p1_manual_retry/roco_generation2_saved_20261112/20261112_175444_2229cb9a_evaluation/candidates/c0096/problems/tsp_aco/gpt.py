import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global triangle-support edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances)
    nonnegative = finite & (distances >= 0.0)
    nonnegative &= ~np.eye(n, dtype=bool)

    if not np.any(nonnegative):
        return result

    # Build an undirected working graph without using endpoint-local scales.
    forward = np.where(nonnegative, distances, np.inf)
    reverse = np.where(nonnegative.T, distances.T, np.inf)
    d = np.minimum(forward, reverse)
    valid = np.isfinite(d)
    np.fill_diagonal(valid, False)
    np.fill_diagonal(d, np.inf)

    if not np.any(valid):
        return result

    finite_edges = d[valid]
    positive_edges = finite_edges[finite_edges > 0.0]
    eps = np.finfo(float).eps

    if positive_edges.size:
        q25, q75 = np.percentile(positive_edges, [25.0, 75.0])
        spread = max(float(q75 - q25), eps)
        center = float(np.median(positive_edges))
    else:
        center = max(float(np.max(finite_edges)), 1.0)
        spread = max(center, eps)

    # Global rather than endpoint-relative attractiveness.
    global_attraction = np.exp(
        -np.clip((d - center) / spread, -30.0, 30.0)
    )
    global_attraction[~valid] = 0.0

    # Count/support short two-edge alternatives for every direct edge.
    # A low two-hop cost relative to the direct edge indicates triangle closure.
    safe_d = np.where(valid, d, np.inf)
    two_hop = np.full((n, n), np.inf, dtype=float)
    support_sum = np.zeros((n, n), dtype=float)
    support_count = np.zeros((n, n), dtype=float)

    for k in range(n):
        via_k = safe_d[:, k, None] + safe_d[None, k, :]
        two_hop = np.minimum(two_hop, via_k)

        threshold = center + 1.5 * spread
        short_legs = (
            np.isfinite(safe_d[:, k, None])
            & np.isfinite(safe_d[None, k, :])
            & (safe_d[:, k, None] <= threshold)
            & (safe_d[None, k, :] <= threshold)
        )
        support_count += short_legs.astype(float)

        finite_via = np.isfinite(via_k)
        support_sum += np.where(finite_via, np.exp(-via_k / max(center, eps)), 0.0)

    np.fill_diagonal(two_hop, np.inf)
    support_count = np.minimum(support_count, max(float(n - 2), 1.0))

    closure_quality = np.divide(
        d,
        np.maximum(two_hop, eps),
        out=np.zeros_like(d),
        where=valid & np.isfinite(two_hop),
    )
    closure_quality = np.clip(closure_quality, 0.0, 2.0)

    support_density = np.divide(
        support_count,
        max(float(n - 2), 1.0),
    )
    support_density = np.clip(support_density, 0.0, 1.0)

    support_mass = np.divide(
        support_sum,
        max(float(n - 2), 1.0),
    )
    support_mass = np.clip(support_mass, 0.0, 1.0)

    triangle_support = (
        0.45 * support_density
        + 0.35 * support_mass
        + 0.20 * np.minimum(closure_quality, 1.0)
    )

    # Prim-style bridge indicator: reward edges useful for connecting
    # otherwise distant components, without constructing a full tour.
    bridge = np.zeros((n, n), dtype=float)
    connected = np.zeros(n, dtype=bool)
    connected[0] = True

    for _ in range(n - 1):
        candidate = np.where(connected[:, None] & (~connected[None, :]) & valid, d, np.inf)
        source, target = np.unravel_index(np.argmin(candidate), candidate.shape)

        if not np.isfinite(candidate[source, target]):
            break

        bridge[source, target] = 1.0
        bridge[target, source] = 1.0
        connected[target] = True

    bridge_bonus = 1.0 + 0.30 * bridge

    # Additive mixing avoids making one signal completely dominate.
    score = (
        0.50 * global_attraction
        + 0.35 * triangle_support
        + 0.15 * global_attraction * triangle_support
    )
    score *= bridge_bonus

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
