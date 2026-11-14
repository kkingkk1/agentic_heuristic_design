import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & ~diagonal

    if not np.any(valid):
        return result

    distances = np.where(valid, np.maximum(raw, 0.0), np.inf)

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    global_temperature = max(0.72 * scale, np.finfo(float).eps)

    row_min = np.min(distances, axis=1)
    row_min = np.where(np.isfinite(row_min), row_min, 0.0)

    local_gap = np.maximum(distances - row_min[:, None], 0.0)
    local_temperature = np.maximum(
        0.38 * scale,
        0.5 * np.median(
            np.where(valid, local_gap, np.nan),
            axis=1,
        ),
    )
    local_temperature = np.where(
        np.isfinite(local_temperature) & (local_temperature > 0.0),
        local_temperature,
        0.38 * scale,
    )

    absolute_affinity = np.exp(
        -np.minimum(distances, 24.0 * global_temperature)
        / global_temperature
    )
    absolute_affinity[~valid] = 0.0

    local_affinity = np.exp(
        -np.minimum(local_gap, 18.0 * local_temperature[:, None])
        / local_temperature[:, None]
    )
    local_affinity[~valid] = 0.0

    ranked_distances = np.where(valid, distances, np.inf)
    order = np.argsort(ranked_distances, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    rank_temperature = max(0.38 * np.sqrt(float(n)), 1.0)
    rank_signal = np.exp(-ranks / rank_temperature)
    rank_signal[~valid] = 0.0

    outgoing_support = np.max(local_affinity, axis=1)
    incoming_support = np.max(local_affinity, axis=0)

    endpoint_potential = np.sqrt(
        outgoing_support[:, None] * incoming_support[None, :]
    )
    endpoint_potential[~valid] = 0.0

    score = (
        0.56 * absolute_affinity
        + 0.25 * local_affinity
        + 0.12 * rank_signal
        + 0.07 * endpoint_potential
    )

    finite_scores = score[valid]
    positive_scores = finite_scores[finite_scores > 0.0]

    if positive_scores.size:
        floor = 0.015 * float(np.median(positive_scores))
        score[valid] = np.maximum(score[valid], floor)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
