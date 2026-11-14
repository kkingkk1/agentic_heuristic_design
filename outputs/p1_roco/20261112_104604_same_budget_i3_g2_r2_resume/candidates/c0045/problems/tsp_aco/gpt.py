import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    inf = np.inf
    eps = np.finfo(float).eps
    work = np.where(valid, distances, inf)

    outgoing_values = np.sort(work, axis=1)
    incoming_values = np.sort(work, axis=0)

    first_outgoing = outgoing_values[:, 0:1]
    first_incoming = incoming_values[0:1, :]

    if n >= 3:
        second_outgoing = outgoing_values[:, 1:2]
        second_incoming = incoming_values[1:2, :]
    else:
        second_outgoing = first_outgoing
        second_incoming = first_incoming

    row_scale = np.where(
        np.isfinite(second_outgoing) & (second_outgoing > 0.0),
        second_outgoing,
        first_outgoing,
    )
    col_scale = np.where(
        np.isfinite(second_incoming) & (second_incoming > 0.0),
        second_incoming,
        first_incoming,
    )

    row_scale = np.maximum(row_scale, eps)
    col_scale = np.maximum(col_scale, eps)

    outgoing_affinity = np.exp(-work / row_scale)
    incoming_affinity = np.exp(-work / col_scale)

    outgoing_affinity[~np.isfinite(outgoing_affinity)] = 0.0
    incoming_affinity[~np.isfinite(incoming_affinity)] = 0.0

    base = np.sqrt(outgoing_affinity * incoming_affinity)

    outgoing_pressure = second_outgoing / (work + second_outgoing + eps)
    incoming_pressure = second_incoming / (work + second_incoming + eps)

    bottleneck = np.sqrt(
        np.clip(outgoing_pressure, 0.0, 1.0)
        * np.clip(incoming_pressure, 0.0, 1.0)
    )
    bottleneck[~np.isfinite(bottleneck)] = 0.0

    if n >= 3:
        two_hop = np.full((n, n), inf, dtype=float)

        for k in range(n):
            candidate = work[:, k:k + 1] + work[k:k + 1, :]
            two_hop = np.minimum(two_hop, candidate)

        np.fill_diagonal(two_hop, inf)

        detour = (two_hop - work) / (two_hop + work + eps)
        detour = np.clip(detour, 0.0, 1.0)
        detour[~np.isfinite(detour)] = 0.0
    else:
        detour = np.zeros((n, n), dtype=float)

    score = 0.65 * base + 0.25 * detour + 0.10 * bottleneck
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
