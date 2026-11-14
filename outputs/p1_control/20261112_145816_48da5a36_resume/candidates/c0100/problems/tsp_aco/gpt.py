import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric TSP edge indicators based on detour advantage."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    positive_values = raw[finite_nonnegative & (raw > 0.0)]

    if positive_values.size:
        scale = float(np.median(positive_values))
        replacement = max(
            float(np.max(positive_values)) * 10.0,
            scale,
            np.finfo(float).eps,
        )
    else:
        scale = 1.0
        replacement = 1.0

    valid = finite_nonnegative & finite_nonnegative.T

    cleaned = np.where(finite_nonnegative, raw, replacement)
    cleaned = 0.5 * (cleaned + cleaned.T)
    cleaned = np.maximum(cleaned, 0.0)
    np.fill_diagonal(cleaned, np.inf)

    eps = np.finfo(float).eps
    scale = max(scale, eps)

    if n >= 3:
        via_i = cleaned[:, :, None] + cleaned[None, :, :]
        via_i = np.transpose(via_i, (0, 2, 1))

        indices = np.arange(n)
        via_i[indices, indices, :] = np.inf
        via_i[:, indices, indices] = np.inf

        best_detour = np.min(via_i, axis=2)
        finite_detour = np.isfinite(best_detour)

        detour_advantage = np.zeros((n, n), dtype=float)
        detour_advantage[finite_detour] = (
            best_detour[finite_detour]
            - cleaned[finite_detour]
        )
        detour_advantage = np.clip(
            detour_advantage / scale,
            -40.0,
            40.0,
        )
    else:
        detour_advantage = np.zeros((n, n), dtype=float)

    shortcut_score = 1.0 / (1.0 + np.exp(-detour_advantage))

    kernel_distance = np.minimum(cleaned / scale, 60.0)
    affinity = np.exp(-kernel_distance)
    affinity[~np.isfinite(affinity)] = 0.0
    np.fill_diagonal(affinity, 0.0)

    endpoint_pressure = np.sum(affinity, axis=1)
    endpoint_pressure = np.maximum(endpoint_pressure, eps)

    edge_affinity = np.sqrt(
        affinity / endpoint_pressure[:, None]
        * affinity.T / endpoint_pressure[None, :]
    )

    exclusivity = edge_affinity / (
        edge_affinity
        + 0.5 * (
            affinity / endpoint_pressure[:, None]
            + affinity.T / endpoint_pressure[None, :]
        )
        + eps
    )

    score = 0.72 * shortcut_score + 0.28 * exclusivity
    score = 0.5 * (score + score.T)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
