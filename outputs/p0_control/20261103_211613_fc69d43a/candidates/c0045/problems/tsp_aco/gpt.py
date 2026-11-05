import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-preference matrix for a minimization TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if np.any(np.isfinite(distances) & (distances < 0.0)):
        raise ValueError("distance_matrix cannot contain negative distances")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    positive = d[valid & (d > 0.0)]
    if positive.size == 0:
        return scores

    global_scale = max(
        float(np.median(positive)),
        np.finfo(float).eps,
    )

    local_scales = np.empty(n, dtype=float)
    for i in range(n):
        values = d[i][valid[i]]
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            local_scales[i] = float(np.median(values))
        else:
            local_scales[i] = global_scale

    local_scales = np.maximum(local_scales, np.finfo(float).eps)
    endpoint_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )

    normalized = d / np.maximum(endpoint_scale, np.finfo(float).eps)
    edge_mask = valid & np.isfinite(normalized)

    inverse_distance = np.zeros_like(d, dtype=float)
    inverse_distance[edge_mask] = 1.0 / (1.0 + normalized[edge_mask])

    competitiveness = np.zeros_like(d, dtype=float)
    for i in range(n):
        alternatives = d[i][valid[i]]
        alternatives = alternatives[
            np.isfinite(alternatives) & (alternatives > 0.0)
        ]
        baseline = (
            float(np.median(alternatives))
            if alternatives.size
            else global_scale
        )
        row_mask = edge_mask[i]
        competitiveness[i, row_mask] = baseline / (
            baseline + d[i, row_mask]
        )

    regret = np.zeros_like(d, dtype=float)
    if n > 2:
        for i in range(n):
            candidates = d[i].copy()
            candidates[~valid[i]] = np.inf
            for j in range(n):
                if not edge_mask[i, j]:
                    continue
                alternatives = np.delete(candidates, j)
                best_alternative = np.min(alternatives)
                if np.isfinite(best_alternative):
                    regret[i, j] = max(
                        best_alternative - d[i, j],
                        0.0,
                    ) / (best_alternative + global_scale)

    bridge = np.zeros_like(d, dtype=float)
    if n > 2:
        two_hop = np.full((n, n), np.inf, dtype=float)
        for k in range(n):
            candidate = d[:, k, None] + d[None, k, :]
            usable = valid[:, k, None] & valid[None, k, :]
            candidate[~usable] = np.inf
            np.minimum(two_hop, candidate, out=two_hop)

        bridge_mask = edge_mask & np.isfinite(two_hop)
        bridge[bridge_mask] = np.maximum(
            two_hop[bridge_mask] - d[bridge_mask],
            0.0,
        ) / (
            two_hop[bridge_mask] + d[bridge_mask] + global_scale
        )

    raw = (
        0.45 * inverse_distance
        + 0.30 * competitiveness
        + 0.15 * regret
        + 0.10 * bridge
    )

    raw[~edge_mask] = 0.0
    scores = 0.5 * (raw + raw.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    scores[~np.isfinite(scores)] = 0.0
    return scores
