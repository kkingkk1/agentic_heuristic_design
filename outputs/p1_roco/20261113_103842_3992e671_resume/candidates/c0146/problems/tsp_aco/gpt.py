import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    distances = distance_matrix.copy()
    distances = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)

    finite = np.isfinite(distances)
    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    nearest = np.min(distances, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]
    scale = float(np.median(finite_nearest))
    scale = max(scale, np.finfo(float).eps)

    normalized = distances / scale
    base = 1.0 / (1.0 + normalized * normalized)
    base[~finite] = 0.0

    rank_signal = np.zeros((n, n), dtype=float)
    for i in range(n):
        valid = np.flatnonzero(finite[i])
        if valid.size == 0:
            continue

        order = valid[np.argsort(distances[i, valid], kind="stable")]
        rank_signal[i, order] = 1.0 - (
            np.arange(order.size, dtype=float) / max(order.size - 1, 1)
        )

    nearest_idx = np.full(n, -1, dtype=int)
    for i in range(n):
        valid = np.flatnonzero(finite[i])
        if valid.size:
            nearest_idx[i] = valid[np.argmin(distances[i, valid])]

    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest_idx):
        if j >= 0 and nearest_idx[j] == i:
            mutual[i, j] = 1.0

    nearest_safe = np.where(np.isfinite(nearest), nearest, scale)
    local_scale = np.sqrt(np.outer(nearest_safe, nearest_safe))
    locality = local_scale / (distances + 0.75 * scale)
    locality = np.clip(locality, 0.0, 1.0)
    locality[~finite] = 0.0

    two_step = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in np.flatnonzero(finite[i]):
            if i == j:
                continue

            via_i = distances[i, :]
            via_j = distances[:, j]
            valid = np.isfinite(via_i) & np.isfinite(via_j)
            valid[i] = False
            valid[j] = False

            if np.any(valid):
                completion = np.min(via_i[valid] + via_j[valid])
                direct = distances[i, j]
                if np.isfinite(completion):
                    two_step[i, j] = np.exp(
                        -0.35 * completion / max(direct + scale, scale)
                    )

    scores = base
    scores *= 0.72 + 0.18 * rank_signal + 0.10 * locality
    scores *= 1.0 + 0.28 * mutual
    scores *= 0.92 + 0.08 * two_step

    scores[~finite] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores, initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores
