import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)

    # Build a symmetric usable distance view from whichever direction is available.
    forward = finite_nonnegative & ~diagonal
    backward = forward.T
    both = forward & backward
    only_forward = forward & ~backward
    only_backward = backward & ~forward

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[both] = 0.5 * (
        distances[both] + distances.T[both]
    )
    symmetric[only_forward] = distances[only_forward]
    symmetric[only_backward] = distances.T[only_backward]

    valid = np.isfinite(symmetric) & ~diagonal
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = symmetric[valid]
    scale = float(np.percentile(finite_values, 35.0))
    if not np.isfinite(scale) or scale <= 0.0:
        positive = finite_values[finite_values > 0.0]
        scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    with np.errstate(over="ignore", invalid="ignore"):
        # Global smooth likelihood with a saturating long-edge penalty.
        normalized = symmetric / scale
        likelihood = 1.0 / (1.0 + np.clip(normalized, 0.0, 1e12) ** 1.7)

    likelihood[~valid] = 0.0

    row_mass = np.sum(likelihood, axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, np.finfo(float).eps)
    directional_choice = likelihood / row_mass

    # Endpoint agreement: an edge is preferred when both endpoints assign
    # substantial probability to selecting it.
    agreement = np.sqrt(
        directional_choice * directional_choice.T
    )

    # Measure concentration of each endpoint's full choice distribution.
    safe_probability = np.clip(directional_choice, 1e-300, 1.0)
    entropy = -np.sum(
        np.where(valid, directional_choice * np.log(safe_probability), 0.0),
        axis=1,
    )
    entropy_scale = np.log(max(n - 1, 2))
    concentration = 1.0 - np.clip(entropy / entropy_scale, 0.0, 1.0)
    endpoint_factor = 0.85 + 0.15 * np.sqrt(
        concentration[:, None] * concentration[None, :]
    )

    scores = agreement * endpoint_factor
    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
