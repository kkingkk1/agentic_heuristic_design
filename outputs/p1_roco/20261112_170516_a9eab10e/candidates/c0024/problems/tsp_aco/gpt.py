import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)

    distances = np.where(np.isfinite(distance_matrix), distance_matrix, np.inf)
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[np.isfinite(distances)]
    if finite_values.size == 0:
        return np.zeros_like(distances, dtype=float)

    positive_values = finite_values[finite_values > 0]
    fallback = float(np.median(positive_values)) if positive_values.size else 1.0
    safe_distances = np.where(
        np.isfinite(distances),
        np.maximum(distances, np.finfo(float).eps),
        fallback * 1e6,
    )

    inverse_distance = 1.0 / safe_distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    k = min(max(2, int(np.sqrt(max(n, 2)))), max(n - 1, 1))
    local_scales = np.partition(safe_distances, k - 1, axis=1)[:, k - 1]
    local_scales = np.maximum(local_scales, np.finfo(float).eps)

    local_compatibility = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    ) / safe_distances
    local_compatibility = np.clip(local_compatibility, 0.0, 1.0)

    reference = np.median(safe_distances, axis=0)
    savings = (
        reference[:, None]
        + reference[None, :]
        - safe_distances
    )
    savings = np.maximum(savings, 0.0)
    savings_scale = np.max(savings)
    if savings_scale > 0:
        savings = savings / savings_scale

    inv_scale = np.max(inverse_distance)
    if inv_scale > 0:
        inverse_distance = inverse_distance / inv_scale

    scores = (
        0.50 * inverse_distance
        + 0.35 * local_compatibility
        + 0.15 * savings
    )

    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores = scores / maximum

    return scores.astype(float, copy=False)
