import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promising indicators for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = distances.copy()
    finite = np.isfinite(d)

    if not np.all(finite | (d == np.inf)):
        raise ValueError("distance_matrix contains invalid values")

    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(finite_values)) + 1.0
    safe_d = np.where(np.isfinite(d), d, fallback)

    # Reciprocal distance with robust scaling.
    positive = safe_d[safe_d > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    inverse_distance = 1.0 / (1.0 + safe_d / max(scale, 1e-12))

    # Endpoint-relative preference: high when an edge is short for both endpoints.
    row_min = np.min(safe_d, axis=1, keepdims=True)
    row_max = np.max(safe_d, axis=1, keepdims=True)
    row_span = np.maximum(row_max - row_min, 1e-12)
    relative_preference = 1.0 - (safe_d - row_min) / row_span

    # Mutual preference rewards edges that are simultaneously attractive
    # from both endpoints, while reducing hub-like long edges.
    mutual_preference = np.sqrt(
        np.clip(relative_preference, 0.0, 1.0)
        * np.clip(relative_preference.T, 0.0, 1.0)
    )

    # Bridge bonus favors edges whose endpoints have complementary local scales.
    local_scale = np.median(safe_d, axis=1)
    local_scale = np.maximum(local_scale, 1e-12)
    scale_balance = np.minimum(local_scale[:, None], local_scale[None, :]) / np.maximum(
        local_scale[:, None], local_scale[None, :]
    )

    score = (
        0.45 * inverse_distance
        + 0.40 * mutual_preference
        + 0.15 * scale_balance
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    valid_off_diagonal = ~np.eye(n, dtype=bool) & np.isfinite(distances)
    if np.any(valid_off_diagonal):
        lo = np.min(score[valid_off_diagonal])
        hi = np.max(score[valid_off_diagonal])
        if hi > lo:
            score = (score - lo) / (hi - lo)

    score[~np.isfinite(distances)] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.asarray(score, dtype=float)
