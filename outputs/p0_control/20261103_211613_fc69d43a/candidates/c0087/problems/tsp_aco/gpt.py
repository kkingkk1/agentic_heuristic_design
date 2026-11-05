import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized heuristic scores for a fixed ant-colony TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    usable = distances[valid]
    if usable.size == 0:
        return scores

    fallback = float(np.median(usable))
    if not np.isfinite(fallback) or fallback <= 0:
        return scores

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n - 1)))))

    def robust_scales(mask: np.ndarray) -> np.ndarray:
        scales = np.full(n, fallback, dtype=float)
        for index in range(n):
            values = distances[index, mask[index]]
            if values.size:
                nearest = np.partition(values, min(k, values.size) - 1)
                scales[index] = float(
                    np.median(nearest[: min(k, values.size)])
                )
        return scales

    outgoing_scale = robust_scales(valid)
    incoming_scale = robust_scales(valid.T)

    safe_distances = np.where(valid, distances, 1.0)

    log_distance = np.log(safe_distances)
    log_outgoing = np.log(outgoing_scale)
    log_incoming = np.log(incoming_scale)

    endpoint_affinity = 0.5 * (
        log_outgoing[:, None]
        + log_incoming[None, :]
        - 2.0 * log_distance
    )

    scale_compatibility = -np.abs(
        log_outgoing[:, None] - log_incoming[None, :]
    )

    log_scores = endpoint_affinity + 0.5 * scale_compatibility
    valid_log_scores = log_scores[valid]

    finite_log_scores = valid_log_scores[np.isfinite(valid_log_scores)]
    if finite_log_scores.size == 0:
        return scores

    maximum = float(np.max(finite_log_scores))
    normalized = np.zeros_like(log_scores, dtype=float)

    normalized[valid] = np.exp(log_scores[valid] - maximum)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    normalized = np.clip(normalized, 0.0, 1.0)

    scores = 0.5 * (normalized + normalized.T)
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
