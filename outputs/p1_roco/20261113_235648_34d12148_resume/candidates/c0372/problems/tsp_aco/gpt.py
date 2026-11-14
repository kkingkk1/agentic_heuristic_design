import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed, distance-dominant edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    valid = offdiag & np.isfinite(raw)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = np.maximum(raw[valid], 0.0)

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps

    # Strong short-edge prior with a robust, global temperature.
    temperature = max(0.42 * scale, eps)
    short_edge = np.exp(
        -np.clip(distances / temperature, 0.0, 40.0)
    )
    short_edge[~valid] = 0.0

    # Stable local competitiveness: rank edges independently by source
    # and destination, while preserving directed evidence.
    masked = np.where(valid, distances, np.inf)

    outgoing_order = np.argsort(masked, axis=1, kind="stable")
    outgoing_rank = np.empty((n, n), dtype=float)
    outgoing_rank[
        np.arange(n)[:, None], outgoing_order
    ] = np.arange(n, dtype=float)[None, :]

    incoming_order = np.argsort(masked.T, axis=1, kind="stable")
    incoming_rank = np.empty((n, n), dtype=float)
    incoming_rank[
        np.arange(n)[:, None], incoming_order
    ] = np.arange(n, dtype=float)[None, :]
    incoming_rank = incoming_rank.T

    rank_temperature = max(1.0, 0.5 * np.sqrt(float(n)))
    outgoing_signal = np.exp(
        -np.clip(outgoing_rank / rank_temperature, 0.0, 40.0)
    )
    incoming_signal = np.exp(
        -np.clip(incoming_rank / rank_temperature, 0.0, 40.0)
    )

    rank_signal = 0.5 * (outgoing_signal + incoming_signal)
    rank_signal[~valid] = 0.0

    # Keep rank pressure mild so it cannot overwhelm distance affinity.
    score = 0.86 * short_edge + 0.14 * rank_signal

    # Preserve a small exploration floor for every valid edge.
    floor = 0.015
    score[valid] = floor + (1.0 - floor) * score[valid]
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
    ).astype(float, copy=False)
