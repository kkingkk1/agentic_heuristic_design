import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally competition-balanced edge indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    diagonal = np.eye(n, dtype=bool)
    finite_offdiag = d[~diagonal & np.isfinite(d)]

    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = finite_offdiag[finite_offdiag > 0.0]
    fallback = float(np.max(positive)) if positive.size else 1.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    if positive.size:
        scale = float(np.median(positive))
        low = float(np.percentile(positive, 25.0))
        high = float(np.percentile(positive, 75.0))
        spread = max(high - low, eps)
    else:
        scale = fallback
        spread = max(fallback, eps)

    scale = max(scale, eps)

    compressed = np.log1p(d / scale)
    affinity = 1.0 / (1.0 + compressed)

    variability = np.median(
        np.abs(d - np.median(d, axis=1, keepdims=True)),
        axis=1,
    )
    temperature = 1.0 + np.clip(variability / spread, 0.0, 3.0)
    temperature_pair = np.sqrt(
        temperature[:, None] * temperature[None, :]
    )

    affinity = affinity ** (1.0 / temperature_pair)
    np.fill_diagonal(affinity, 0.0)

    score = np.maximum(affinity, 0.0)

    for _ in range(32):
        row_mass = np.sum(score, axis=1)
        row_mass = np.maximum(row_mass, eps)
        score = score / np.sqrt(
            row_mass[:, None] * row_mass[None, :]
        )
        score = 0.5 * (score + score.T)
        np.fill_diagonal(score, 0.0)

    endpoint_mass = np.sum(score, axis=1)
    endpoint_mass = np.maximum(endpoint_mass, eps)

    competition = score / np.sqrt(
        endpoint_mass[:, None] * endpoint_mass[None, :]
    )

    edge_scale = np.sqrt(
        affinity * competition
    )
    score = 0.55 * competition + 0.45 * edge_scale

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
