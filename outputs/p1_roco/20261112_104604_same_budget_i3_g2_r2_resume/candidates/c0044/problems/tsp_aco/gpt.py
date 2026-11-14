import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, distances, np.inf)
    eps = np.finfo(float).eps

    # Direct-edge advantage over the best valid two-hop alternative.
    if n >= 3:
        two_hop = np.full((n, n), np.inf, dtype=float)

        for k in range(n):
            left = work[:, k:k + 1]
            right = work[k:k + 1, :]
            two_hop = np.minimum(two_hop, left + right)

        np.fill_diagonal(two_hop, np.inf)

        finite_detour = valid & np.isfinite(two_hop)
        detour_advantage = np.zeros((n, n), dtype=float)

        numerator = two_hop - distances
        denominator = two_hop + distances + eps

        detour_advantage[finite_detour] = (
            numerator[finite_detour] / denominator[finite_detour]
        )
        detour_advantage = np.clip(detour_advantage, 0.0, 1.0)
    else:
        detour_advantage = np.zeros((n, n), dtype=float)

    # Endpoint bottleneck pressure using only valid outgoing/incoming edges.
    outgoing_sorted = np.sort(work, axis=1)
    incoming_sorted = np.sort(work, axis=0)

    if n >= 3:
        second_outgoing = outgoing_sorted[:, 1:2]
        second_incoming = incoming_sorted[1:2, :]

        finite_outgoing = np.isfinite(second_outgoing)
        finite_incoming = np.isfinite(second_incoming)

        outgoing_pressure = np.zeros((n, n), dtype=float)
        incoming_pressure = np.zeros((n, n), dtype=float)

        outgoing_denominator = distances + second_outgoing + eps
        incoming_denominator = distances + second_incoming + eps

        outgoing_pressure[valid & finite_outgoing] = (
            second_outgoing.repeat(n, axis=1)[valid & finite_outgoing]
            / outgoing_denominator[valid & finite_outgoing]
        )
        incoming_pressure[valid & finite_incoming] = (
            second_incoming.repeat(n, axis=0)[valid & finite_incoming]
            / incoming_denominator[valid & finite_incoming]
        )

        bottleneck_score = np.sqrt(
            np.clip(outgoing_pressure, 0.0, 1.0)
            * np.clip(incoming_pressure, 0.0, 1.0)
        )
    else:
        bottleneck_score = np.zeros((n, n), dtype=float)

    # Robust local affinity. Row and column scales are computed independently
    # from valid edges, so invalid sentinels never affect the statistics.
    row_scale = np.full(n, np.nan, dtype=float)
    col_scale = np.full(n, np.nan, dtype=float)

    for i in range(n):
        row_values = distances[i, valid[i]]
        if row_values.size:
            row_scale[i] = np.median(row_values)

    for j in range(n):
        col_values = distances[valid[:, j], j]
        if col_values.size:
            col_scale[j] = np.median(col_values)

    safe_row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0.0),
        row_scale,
        1.0,
    )
    safe_col_scale = np.where(
        np.isfinite(col_scale) & (col_scale > 0.0),
        col_scale,
        1.0,
    )

    row_affinity = np.zeros((n, n), dtype=float)
    col_affinity = np.zeros((n, n), dtype=float)

    row_ratio = distances / (safe_row_scale[:, None] + eps)
    col_ratio = distances / (safe_col_scale[None, :] + eps)

    row_affinity[valid] = 1.0 / (1.0 + np.maximum(row_ratio[valid], 0.0))
    col_affinity[valid] = 1.0 / (1.0 + np.maximum(col_ratio[valid], 0.0))

    local_affinity = np.sqrt(row_affinity * col_affinity)

    # Primary structural score. Local affinity remains deliberately modest.
    structural_score = (
        0.68 * detour_advantage
        + 0.22 * bottleneck_score
        + 0.10 * local_affinity
    )

    # Modest global regularization: retain a bounded closeness signal without
    # replacing the directed structural evidence.
    valid_values = distances[valid]
    global_scale = np.median(valid_values) if valid_values.size else 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    global_affinity = np.zeros((n, n), dtype=float)
    global_ratio = distances / (global_scale + eps)
    global_affinity[valid] = 1.0 / (1.0 + np.maximum(global_ratio[valid], 0.0))

    score = 0.90 * structural_score + 0.10 * global_affinity
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    finite_score = np.isfinite(score)
    score[~finite_score] = 0.0

    maximum = np.max(score)
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
