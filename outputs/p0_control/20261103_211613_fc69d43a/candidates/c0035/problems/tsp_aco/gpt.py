import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge scores for a fixed minimization TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    if np.any(np.isfinite(distances) & (distances < 0.0)):
        raise ValueError("distance_matrix cannot contain negative distances")

    valid = np.isfinite(distances)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return scores

    global_scale = max(
        float(np.median(positive)),
        np.finfo(float).eps,
    )

    safe = np.where(valid, distances, global_scale * 1e6)
    safe = np.maximum(safe, 0.0)

    row_scales = np.full(n, global_scale, dtype=float)
    for i in range(n):
        values = distances[i, valid[i] & (distances[i] > 0.0)]
        if values.size:
            row_scales[i] = max(
                float(np.median(values)),
                np.finfo(float).eps,
            )

    scaled = safe / row_scales[:, None]
    preference = np.exp(-np.minimum(scaled, 700.0))
    preference[~valid] = 0.0

    row_totals = np.sum(preference, axis=1, keepdims=True)
    row_totals = np.maximum(row_totals, np.finfo(float).eps)
    preference /= row_totals

    direct = preference.copy()

    ranks = np.zeros_like(distances, dtype=float)
    for i in range(n):
        row_valid = valid[i]
        count = int(np.count_nonzero(row_valid))
        if count == 0:
            continue
        values = distances[i, row_valid]
        ranks[i, row_valid] = (
            np.sum(values[:, None] <= values[None, :], axis=0) - 1.0
        ) / max(count - 1, 1)

    local_competition = 1.0 - 0.5 * (
        ranks + ranks.T
    )
    local_competition = np.clip(local_competition, 0.0, 1.0)

    norms = np.sqrt(np.sum(preference * preference, axis=1))
    norms = np.maximum(norms, np.finfo(float).eps)
    similarity = (preference @ preference.T) / (
        norms[:, None] * norms[None, :]
    )
    similarity = np.clip(similarity, 0.0, 1.0)

    entropy = -np.sum(
        np.where(preference > 0.0, preference * np.log(preference), 0.0),
        axis=1,
    )
    entropy_scale = max(np.log(max(n - 1, 2)), np.finfo(float).eps)
    entropy = np.clip(entropy / entropy_scale, 0.0, 1.0)
    stability = np.sqrt(entropy[:, None] * entropy[None, :])

    scores = (
        (0.55 * direct + 0.45 * local_competition)
        * (0.35 + 0.65 * similarity)
        * (0.5 + 0.5 * stability)
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    scores[~np.isfinite(scores)] = 0.0
    return scores
