import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric spectral edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)
    valid = finite & finite.T

    if not np.any(valid):
        return result

    with np.errstate(all="ignore"):
        d = 0.5 * (distances + distances.T)

    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    values = d[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    elif values.size:
        scale = float(np.max(values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    # Convert distance to a bounded, scale-invariant conductance.
    relative_distance = np.divide(
        d,
        scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    conductance = np.divide(
        1.0,
        1.0 + np.maximum(relative_distance, 0.0),
        out=np.zeros_like(d, dtype=float),
        where=valid,
    )
    conductance[~valid] = 0.0
    np.fill_diagonal(conductance, 0.0)

    weighted_degree = np.sum(conductance, axis=1)
    degree_reference = float(
        np.median(weighted_degree[weighted_degree > 0.0])
    )
    if not np.isfinite(degree_reference) or degree_reference <= 0.0:
        degree_reference = 1.0

    # Mildly protect endpoints with unusually little usable connectivity.
    endpoint_need = np.sqrt(
        degree_reference
        / np.maximum(weighted_degree, eps)
    )
    endpoint_need = np.clip(endpoint_need, 0.75, 1.35)
    endpoint_factor = np.sqrt(
        endpoint_need[:, None] * endpoint_need[None, :]
    )

    # Find connected components of the valid graph.
    adjacency = valid.copy()
    unseen = np.ones(n, dtype=bool)
    components = []

    for start in range(n):
        if not unseen[start]:
            continue

        stack = [start]
        unseen[start] = False
        members = []

        while stack:
            node = stack.pop()
            members.append(node)
            neighbors = np.flatnonzero(adjacency[node] & unseen)
            if neighbors.size:
                unseen[neighbors] = False
                stack.extend(neighbors.tolist())

        components.append(np.asarray(members, dtype=int))

    leverage = np.zeros((n, n), dtype=float)

    # Effective resistance gives a global, graph-level edge importance signal.
    for component in components:
        if component.size < 2:
            continue

        local_conductance = conductance[np.ix_(component, component)]
        local_degree = np.sum(local_conductance, axis=1)

        laplacian = -local_conductance.copy()
        np.fill_diagonal(laplacian, local_degree)

        try:
            laplacian_pinv = np.linalg.pinv(
                laplacian,
                rcond=1e-10,
            )
        except np.linalg.LinAlgError:
            laplacian_pinv = np.zeros_like(laplacian)

        diagonal = np.diag(laplacian_pinv)
        resistance = (
            diagonal[:, None]
            + diagonal[None, :]
            - 2.0 * laplacian_pinv
        )
        resistance = np.maximum(resistance, 0.0)
        resistance[~np.isfinite(resistance)] = 0.0

        local_leverage = local_conductance * resistance
        local_leverage = np.clip(local_leverage, 0.0, 1.0)
        np.fill_diagonal(local_leverage, 0.0)

        leverage[np.ix_(component, component)] = local_leverage

    valid_leverage = leverage[valid]
    leverage_reference = float(
        np.percentile(valid_leverage, 75)
    ) if valid_leverage.size else 1.0

    if not np.isfinite(leverage_reference) or leverage_reference <= eps:
        leverage_reference = 1.0

    normalized_leverage = np.clip(
        leverage / leverage_reference,
        0.0,
        1.5,
    )

    # Reward globally important edges, but prevent this term from
    # overwhelming the distance preference.
    bridge_factor = 1.0 + 0.24 * normalized_leverage

    # Penalize edges that are disproportionately long relative to
    # the conductance available at both endpoints.
    endpoint_capacity = np.sqrt(
        np.maximum(weighted_degree[:, None], eps)
        * np.maximum(weighted_degree[None, :], eps)
    )
    relative_conductance = np.divide(
        conductance,
        endpoint_capacity,
        out=np.zeros_like(conductance),
        where=endpoint_capacity > 0.0,
    )

    positive_relative = relative_conductance[valid]
    reference = float(np.median(positive_relative)) \
        if positive_relative.size else 1.0

    if not np.isfinite(reference) or reference <= eps:
        reference = 1.0

    local_fit = np.clip(
        np.sqrt(
            np.maximum(relative_conductance, 0.0)
            / reference
        ),
        0.35,
        1.5,
    )

    scores = (
        np.power(np.maximum(conductance, 0.0), 0.85)
        * (0.70 + 0.30 * local_fit)
        * (0.88 + 0.12 * endpoint_factor)
        * bridge_factor
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

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
