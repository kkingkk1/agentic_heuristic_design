import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)
    valid = finite & finite.T

    if not np.any(valid):
        return result

    symmetric_distances = 0.5 * (distances + distances.T)
    symmetric_distances[~valid] = np.inf
    np.fill_diagonal(symmetric_distances, np.inf)

    finite_values = symmetric_distances[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        temperature = float(np.median(positive_values))
    elif finite_values.size:
        temperature = float(np.max(finite_values))
    else:
        temperature = 1.0

    if not np.isfinite(temperature) or temperature <= 0.0:
        temperature = 1.0
    temperature = max(temperature, eps)

    normalized = np.divide(
        symmetric_distances,
        temperature,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    affinity = np.exp(-np.clip(normalized, 0.0, 50.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity = np.maximum(affinity, 0.0)
    row_mass = affinity.sum(axis=1)

    if not np.any(row_mass > 0.0):
        return result

    # Symmetric diagonal scaling approximately equalizes available edge mass.
    scale = np.divide(
        1.0,
        np.sqrt(np.maximum(row_mass, eps)),
        out=np.zeros(n, dtype=float),
        where=row_mass > 0.0,
    )

    for _ in range(18):
        balanced_mass = affinity @ scale
        update = np.divide(
            1.0,
            np.maximum(balanced_mass, eps),
            out=np.zeros(n, dtype=float),
            where=balanced_mass > 0.0,
        )
        scale *= np.sqrt(update)

        scale = np.nan_to_num(
            scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        maximum_scale = float(np.max(scale))
        if maximum_scale > 0.0 and np.isfinite(maximum_scale):
            scale /= maximum_scale

    balanced = affinity * scale[:, None] * scale[None, :]
    balanced[~valid] = 0.0
    np.fill_diagonal(balanced, 0.0)

    # Effective-resistance signal estimates global connective usefulness.
    weighted_degree = balanced.sum(axis=1)
    laplacian = np.diag(weighted_degree) - balanced

    try:
        laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    except TypeError:
        laplacian_pinv = np.linalg.pinv(laplacian)

    resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance[~valid] = 0.0
    np.fill_diagonal(resistance, 0.0)

    leverage = balanced * resistance
    positive_leverage = leverage[valid & (leverage > 0.0)]

    if positive_leverage.size:
        leverage_scale = float(np.median(positive_leverage))
    else:
        leverage_scale = 1.0

    leverage_scale = max(leverage_scale, eps)

    global_connectivity = 1.0 + 0.16 * np.tanh(
        leverage / leverage_scale
    )
    global_connectivity[~valid] = 1.0

    scores = balanced * global_connectivity
    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
