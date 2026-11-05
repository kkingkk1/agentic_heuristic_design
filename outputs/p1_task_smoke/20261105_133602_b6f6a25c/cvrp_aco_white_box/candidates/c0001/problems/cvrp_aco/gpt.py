import numpy as np


def heuristics_v2(
    distance_matrix: np.ndarray,
    coordinates: np.ndarray,
    demands: np.ndarray,
    capacity: int,
) -> np.ndarray:
    """Return CVRP edge-selection priors with depot-aware savings and capacity cues."""
    distances = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    demand = np.asarray(demands, dtype=float).reshape(-1)

    n = distances.shape[0]
    if distances.ndim != 2 or distances.shape[1] != n:
        raise ValueError("distance_matrix must be a square matrix")
    if coords.shape != (n, 2):
        raise ValueError("coordinates must have shape (n, 2)")
    if demand.shape[0] != n:
        raise ValueError("demands must have length n")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if n == 0:
        return np.zeros_like(distances, dtype=float)

    d = np.maximum((distances + distances.T) * 0.5, 0.0)
    np.fill_diagonal(d, 0.0)

    depot_distance = d[0].copy()
    positive_edges = d[(d > 0) & np.isfinite(d)]
    distance_scale = float(np.median(positive_edges)) if positive_edges.size else 1.0
    distance_scale = max(distance_scale, np.finfo(float).eps)

    inverse_distance = 1.0 / (1.0 + d / distance_scale)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    depot_i = depot_distance[:, None]
    depot_j = depot_distance[None, :]
    savings = depot_i + depot_j - d
    savings = np.maximum(savings, 0.0)
    savings_scale = float(np.max(savings)) if np.any(savings > 0) else 1.0
    savings /= max(savings_scale, np.finfo(float).eps)

    demand_i = demand[:, None]
    demand_j = demand[None, :]
    pair_load = demand_i + demand_j
    capacity_ratio = np.clip(pair_load / float(capacity), 0.0, None)
    compatibility = np.exp(-1.5 * capacity_ratio)
    compatibility[pair_load > capacity] *= 0.08

    centered = coords - coords[0]
    radii = np.linalg.norm(centered, axis=1)
    denom = np.maximum(radii[:, None] * radii[None, :], np.finfo(float).eps)
    cosine = np.clip((centered @ centered.T) / denom, -1.0, 1.0)
    angular_alignment = 0.5 * (cosine + 1.0)
    angular_alignment[0, :] = 1.0
    angular_alignment[:, 0] = 1.0

    customer_priority = np.clip(demand / float(capacity), 0.0, 1.0)
    depot_priority = 0.65 * inverse_distance + 0.35 * customer_priority
    depot_priority[0] = 0.0

    score = (
        0.34 * inverse_distance
        + 0.32 * savings
        + 0.20 * compatibility
        + 0.14 * angular_alignment
    )
    score = 0.5 * (score + score.T)

    score[0, :] = depot_priority
    score[:, 0] = depot_priority
    score[0, 0] = 0.0
    np.fill_diagonal(score, 0.0)

    finite_scores = np.isfinite(score)
    score[~finite_scores] = 0.0
    return score.astype(float, copy=False)
