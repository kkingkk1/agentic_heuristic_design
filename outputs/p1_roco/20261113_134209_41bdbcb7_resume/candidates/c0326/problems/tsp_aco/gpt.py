import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    forward = np.where(
        np.isfinite(distances) & (distances >= 0.0) & ~diagonal,
        distances,
        np.inf,
    )
    backward = forward.T

    distance = np.minimum(forward, backward)
    valid = np.isfinite(distance) & ~diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    normalized = np.divide(
        distance,
        scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.clip(normalized[valid] ** 1.35, 0.0, 60.0)
    )

    strength = affinity.sum(axis=1)
    strength = np.maximum(strength, eps)

    balanced_affinity = affinity / np.sqrt(
        strength[:, None] * strength[None, :]
    )

    diffusion = balanced_affinity @ balanced_affinity
    diffusion = np.maximum(diffusion, 0.0)

    support_scale = np.maximum(
        np.sqrt(
            diffusion.sum(axis=1)[:, None]
            * diffusion.sum(axis=1)[None, :]
        ),
        eps,
    )
    normalized_support = np.divide(
        diffusion,
        support_scale,
        out=np.zeros_like(diffusion),
        where=support_scale > 0.0,
    )

    two_hop_distance = np.full((n, n), np.inf, dtype=float)

    for intermediary in range(n):
        left = distance[:, intermediary][:, None]
        right = distance[intermediary, :][None, :]
        candidate = left + right
        two_hop_distance = np.minimum(two_hop_distance, candidate)

    detour_ratio = np.divide(
        two_hop_distance,
        np.maximum(distance + scale, eps),
        out=np.ones((n, n), dtype=float),
        where=valid & np.isfinite(two_hop_distance),
    )
    direct_advantage = 1.0 / (
        1.0
        + 0.55 * np.clip(detour_ratio - 1.0, 0.0, 8.0)
    )

    closure = 1.0 + 0.42 * np.clip(
        normalized_support,
        0.0,
        1.0,
    )

    mst_edge = np.zeros((n, n), dtype=float)
    connected = np.zeros(n, dtype=bool)

    for _ in range(n):
        best_vertex = -1
        best_key = np.inf

        for vertex in range(n):
            if not connected[vertex]:
                finite_incident = valid[vertex] & ~connected
                if np.any(finite_incident):
                    candidate_key = np.min(
                        np.where(
                            finite_incident,
                            distance[vertex],
                            np.inf,
                        )
                    )
                    if candidate_key < best_key:
                        best_key = candidate_key
                        best_vertex = vertex

        if best_vertex < 0:
            for vertex in range(n):
                if not connected[vertex]:
                    connected[vertex] = True
                    best_vertex = vertex
                    break
            if best_vertex < 0:
                break

        connected[best_vertex] = True

        neighboring = valid[best_vertex] & connected
        neighboring[best_vertex] = False

        if np.any(neighboring):
            parent = np.where(
                neighboring,
                distance[best_vertex],
                np.inf,
            )
            other = int(np.argmin(parent))
            if np.isfinite(parent[other]):
                mst_edge[best_vertex, other] = 1.0
                mst_edge[other, best_vertex] = 1.0

        remaining = (~connected) & valid[best_vertex]
        if np.any(remaining):
            next_vertex = int(
                np.argmin(
                    np.where(
                        remaining,
                        distance[best_vertex],
                        np.inf,
                    )
                )
            )
            if np.isfinite(distance[best_vertex, next_vertex]):
                connected[next_vertex] = False

    edge_balance = 1.0 / np.sqrt(
        (1.0 + strength[:, None])
        * (1.0 + strength[None, :])
    )
    edge_balance /= max(float(np.max(edge_balance)), eps)

    scores = (
        affinity
        * (0.58 + 0.42 * edge_balance)
        * closure
        * direct_advantage
        * (1.0 + 0.30 * mst_edge)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
