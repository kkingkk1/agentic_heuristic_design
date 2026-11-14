import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif finite_values.size:
        scale = float(np.max(finite_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    normalized = np.divide(
        distances,
        scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    conductance = np.zeros_like(distances, dtype=float)
    conductance[valid] = 1.0 / (1.0 + normalized[valid])

    outgoing_access = conductance.sum(axis=1)
    incoming_access = conductance.sum(axis=0)

    outgoing_max = max(float(np.max(outgoing_access)), 1.0)
    incoming_max = max(float(np.max(incoming_access)), 1.0)

    destination_access = np.divide(
        outgoing_access,
        outgoing_max,
    )
    source_access = np.divide(
        incoming_access,
        incoming_max,
    )

    continuation = np.sqrt(
        destination_access[None, :]
        * source_access[:, None]
    )

    scores = conductance * (0.65 + 0.35 * continuation)
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = scores.sum(axis=1, keepdims=True)

    return np.divide(
        scores,
        row_totals,
        out=np.zeros_like(scores),
        where=row_totals > 0.0,
    )
