import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally spectral edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    positive = d[valid & (d > 0.0)]

    eps = np.finfo(float).eps
    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    normalized = np.zeros_like(d, dtype=float)
    normalized[valid] = d[valid] / scale

    # Heat-kernel affinities create a dense global weighted graph while
    # preserving stronger preference for shorter valid edges.
    affinity = np.zeros_like(d, dtype=float)
    affinity[valid] = np.exp(-np.clip(normalized[valid], 0.0, 60.0))

    degree = affinity.sum(axis=1)
    active = degree > 0.0

    if not np.any(active):
        return scores

    laplacian = -affinity.copy()
    np.fill_diagonal(laplacian, degree)

    # The pseudoinverse provides global connectivity information rather than
    # relying on a fixed local neighborhood.
    try:
        laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    except TypeError:
        laplacian_pinv = np.linalg.pinv(laplacian)

    diagonal = np.diag(laplacian_pinv)
    resistance = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance = np.nan_to_num(
        resistance,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    leverage = affinity * resistance
    leverage_values = leverage[valid]
    leverage_scale = (
        float(np.median(leverage_values))
        if leverage_values.size
        else 1.0
    )
    leverage_scale = max(leverage_scale, eps)

    # Penalize spectrally fragile edges, but retain a positive exploratory
    # contribution for every valid edge.
    stability = np.exp(
        -np.clip(0.55 * leverage / leverage_scale, 0.0, 60.0)
    )

    degree_scale = np.sqrt(
        np.maximum(degree[:, None] * degree[None, :], eps)
    )
    balance = 1.0 / np.sqrt(
        1.0 + degree_scale / max(float(np.median(degree[active])), eps)
    )

    global_affinity = np.sqrt(
        np.maximum(affinity * affinity.T, 0.0)
    )

    scores = (
        global_affinity
        * (0.70 + 0.30 * stability)
        * (0.80 + 0.20 * balance)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
