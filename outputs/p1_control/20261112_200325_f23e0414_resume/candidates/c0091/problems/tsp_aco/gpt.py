import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite_positive = distances[
        np.isfinite(distances) & (distances > 0.0)
    ]

    if finite_positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(finite_positive))
    fallback = max(fallback, eps)

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    np.fill_diagonal(d, fallback)

    positive = d[valid & (d > 0.0)]
    reference = (
        float(np.median(positive))
        if positive.size
        else fallback
    )
    reference = max(reference, eps)

    transformed = np.log1p(d / reference)

    affinity = 1.0 / (1.0 + transformed)
    np.fill_diagonal(affinity, 0.0)
    affinity[~valid] = 0.0

    row_mass = np.sum(affinity, axis=1, keepdims=True)
    transition = affinity / np.maximum(row_mass, eps)

    reciprocal_flow = np.sqrt(
        transition * transition.T
    )

    shared_support = transition @ transition.T
    support_scale = np.maximum(
        np.max(shared_support),
        eps,
    )
    shared_support = shared_support / support_scale

    expected_outbound = transition @ d
    expected_inbound = d @ transition.T
    expected_two_step = (
        expected_outbound[:, None]
        + expected_inbound[None, :]
    )

    advantage = (
        expected_two_step - d
    ) / np.maximum(
        expected_two_step + d + reference,
        eps,
    )
    advantage = np.clip(advantage, -1.0, 1.0)
    direct_gain = 0.5 * (advantage + 1.0)

    row_variance = np.sum(
        transition * (d - expected_outbound[:, None]) ** 2,
        axis=1,
    )
    uncertainty = np.sqrt(np.maximum(row_variance, 0.0))
    uncertainty_scale = np.maximum(
        uncertainty[:, None] + uncertainty[None, :],
        eps,
    )

    stability = np.exp(
        -np.clip(
            np.abs(d - expected_two_step)
            / (uncertainty_scale + reference),
            0.0,
            40.0,
        )
    )

    score = (
        0.35 * direct_gain
        + 0.30 * reciprocal_flow
        + 0.20 * shared_support
        + 0.15 * stability
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
