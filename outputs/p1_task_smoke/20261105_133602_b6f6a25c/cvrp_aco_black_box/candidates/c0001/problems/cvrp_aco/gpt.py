import numpy as np

def heuristics_v2(edge_attr: np.ndarray, node_attr: np.ndarray) -> np.ndarray:
    edge = np.asarray(edge_attr, dtype=float)
    nodes = np.asarray(node_attr, dtype=float).reshape(-1)

    if edge.ndim != 2 or edge.shape[0] != edge.shape[1]:
        raise ValueError("edge_attr must be a square matrix")
    n = edge.shape[0]
    if nodes.shape[0] != n:
        raise ValueError("node_attr must have the same length as edge_attr")

    if n == 0:
        return np.zeros_like(edge, dtype=float)

    finite_edge = np.isfinite(edge)
    safe_edge = np.where(finite_edge, edge, np.nan)

    finite_values = safe_edge[np.isfinite(safe_edge)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fill_value = np.nanmedian(finite_values)
    safe_edge = np.where(np.isfinite(safe_edge), safe_edge, fill_value)

    scale = np.nanpercentile(np.abs(finite_values), 75)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    cost = np.maximum(safe_edge, 0.0)
    cost = cost / scale

    inverse_cost = 1.0 / (1.0 + cost)
    inverse_cost = np.clip(inverse_cost, 0.0, 1.0)

    demand = np.where(np.isfinite(nodes), np.abs(nodes), 0.0)
    demand_scale = np.percentile(demand, 75)
    if not np.isfinite(demand_scale) or demand_scale <= 1e-12:
        demand_scale = 1.0
    demand = np.clip(demand / demand_scale, 0.0, 3.0)

    depot_cost = cost[0].copy()
    depot_cost[0] = 0.0

    savings = (
        depot_cost[:, None]
        + depot_cost[None, :]
        - cost
    )
    savings_scale = np.percentile(np.abs(savings[1:, 1:]), 75) if n > 1 else 1.0
    if not np.isfinite(savings_scale) or savings_scale <= 1e-12:
        savings_scale = 1.0
    savings = np.tanh(savings / savings_scale)
    savings = np.maximum(savings, 0.0)

    demand_similarity = np.exp(
        -np.abs(demand[:, None] - demand[None, :])
    )

    radial_quality = 1.0 / (
        1.0 + 0.5 * (depot_cost[:, None] + depot_cost[None, :])
    )

    customer_score = (
        0.45 * inverse_cost
        + 0.35 * savings
        + 0.10 * demand_similarity
        + 0.10 * radial_quality
    )

    depot_score = inverse_cost
    score = customer_score.copy()
    score[0, :] = depot_score[0, :]
    score[:, 0] = depot_score[:, 0]

    score = 0.5 * (score + score.T)
    score[~finite_edge] = 0.0
    np.fill_diagonal(score, 0.0)

    score -= np.nanmin(score)
    maximum = np.nanmax(score)
    if np.isfinite(maximum) and maximum > 1e-12:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
