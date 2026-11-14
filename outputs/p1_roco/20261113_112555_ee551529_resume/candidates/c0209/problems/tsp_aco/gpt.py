import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally calibrated, detour-aware TSP edge priors."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(finite):
        return result

    symmetric = np.full((n, n), np.inf, dtype=float)
    forward = np.where(finite, distances, np.inf)
    symmetric = np.minimum(forward, forward.T)
    np.fill_diagonal(symmetric, np.inf)

    valid = np.isfinite(symmetric) & ~diagonal
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = symmetric[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.percentile(positive_values, 60.0))
    else:
        scale = float(np.percentile(finite_values, 60.0))

    if not np.isfinite(scale) or scale <= eps:
        scale = max(float(np.max(finite_values)), 1.0)

    scale = max(scale, eps)

    normalized = np.divide(
        symmetric,
        scale,
        out=np.full_like(symmetric, np.inf),
        where=np.isfinite(symmetric),
    )

    distance_priority = 1.0 / (1.0 + np.clip(normalized, 0.0, 50.0) ** 2)

    finite_flat = finite_values[np.isfinite(finite_values)]
    sorted_values = np.sort(finite_flat)

    edge_position = np.searchsorted(
        sorted_values,
        np.where(valid, symmetric, sorted_values[-1]),
        side="right",
    )
    edge_percentile = edge_position / max(float(sorted_values.size), 1.0)

    global_contrast = 1.0 + 0.24 * np.clip(
        1.0 - 2.0 * edge_percentile,
        -1.0,
        1.0,
    )

    if n >= 3:
        left = symmetric[:, :, None]
        right = symmetric[None, :, :]

        two_hop = left + right
        hop_valid = (
            np.isfinite(left)
            & np.isfinite(right)
        )

        node_indices = np.arange(n)
        hop_valid[:, :, node_indices] = False

        two_hop = np.where(hop_valid, two_hop, np.inf)
        best_detour = np.min(two_hop, axis=2)

        detour_ratio = np.divide(
            best_detour,
            symmetric + scale,
            out=np.full_like(symmetric, np.inf),
            where=valid,
        )

        detour_exclusivity = 1.0 - np.exp(
            -np.clip(detour_ratio, 0.0, 20.0)
        )

        replacement_pressure = np.divide(
            symmetric,
            best_detour,
            out=np.zeros_like(symmetric),
            where=np.isfinite(best_detour) & (best_detour > eps),
        )

        replacement_penalty = 1.0 - 0.20 * np.clip(
            replacement_pressure,
            0.0,
            1.0,
        )

        detour_term = (
            0.72
            + 0.28 * detour_exclusivity
        ) * replacement_penalty
    else:
        detour_term = np.ones((n, n), dtype=float)

    scores = (
        distance_priority
        * global_contrast
        * detour_term
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
