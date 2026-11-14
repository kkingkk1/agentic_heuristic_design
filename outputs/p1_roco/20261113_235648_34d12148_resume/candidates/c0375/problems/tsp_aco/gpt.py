import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded edge priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~np.eye(n, dtype=bool)
    )

    if not np.any(valid):
        return result

    valid_values = raw[valid]
    positive_values = valid_values[valid_values > 0.0]

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    normalized_distance = np.zeros((n, n), dtype=float)
    normalized_distance[valid] = raw[valid] / scale

    reciprocal = np.zeros((n, n), dtype=float)
    reciprocal[valid] = 1.0 / (
        1.0 + normalized_distance[valid]
    )

    temperature = max(0.82, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(normalized_distance[valid] / temperature, 60.0)
    )

    def rank_signal(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.where(mask, values, np.inf)
        order = np.argsort(masked, axis=1, kind="stable")
        ranks = np.empty((n, n), dtype=float)
        ranks[np.arange(n)[:, None], order] = np.arange(n)

        counts = mask.sum(axis=1).astype(float)
        rank_scale = np.maximum(
            1.0,
            0.82 * np.sqrt(np.maximum(counts, 1.0)),
        )

        signal = np.exp(-ranks / rank_scale[:, None])
        signal[~mask] = 0.0
        return signal

    outgoing_rank = rank_signal(raw, valid)
    incoming_rank = rank_signal(raw.T, valid.T).T

    local_signal = np.sqrt(
        np.maximum(outgoing_rank * incoming_rank, 0.0)
    )

    score = (
        0.58 * reciprocal
        + 0.27 * affinity
        + 0.15 * local_signal
    )

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    score /= maximum

    exploration_floor = 0.015
    score[valid] = (
        exploration_floor
        + (1.0 - exploration_floor) * score[valid]
    )

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
