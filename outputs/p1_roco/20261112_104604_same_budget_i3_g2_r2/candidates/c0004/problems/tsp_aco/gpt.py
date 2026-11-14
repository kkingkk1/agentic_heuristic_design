import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, distances, np.inf)
    eps = np.finfo(float).eps

    if n >= 3:
        two_hop = np.full((n, n), np.inf, dtype=float)

        for k in range(n):
            candidate = work[:, k:k + 1] + work[k:k + 1, :]
            two_hop = np.minimum(two_hop, candidate)

        np.fill_diagonal(two_hop, np.inf)
        detour_advantage = (two_hop - work) / (two_hop + work + eps)
        detour_advantage = np.clip(detour_advantage, 0.0, 1.0)
        detour_advantage[~np.isfinite(detour_advantage)] = 0.0
    else:
        detour_advantage = np.zeros((n, n), dtype=float)

    outgoing_sorted = np.sort(work, axis=1)
    incoming_sorted = np.sort(work, axis=0)

    second_outgoing = outgoing_sorted[:, 1:2]
    second_incoming = incoming_sorted[1:2, :]

    outgoing_pressure = second_outgoing / (work + second_outgoing + eps)
    incoming_pressure = second_incoming / (work + second_incoming + eps)

    bottleneck_score = np.sqrt(
        np.clip(outgoing_pressure, 0.0, 1.0)
        * np.clip(incoming_pressure, 0.0, 1.0)
    )

    score = 0.75 * detour_advantage + 0.25 * bottleneck_score
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
