import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return short-edge-dominant structural priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & np.isfinite(raw.T)
        & (raw.T >= 0.0)
        & ~diagonal
    )

    if not np.any(valid):
        return result

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    valid_values = distances[valid]
    positive_values = valid_values[valid_values > 0.0]

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    temperature = max(0.80 * scale, eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / temperature, 60.0)
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + eps)

    inverse_max = float(inverse.max())
    if np.isfinite(inverse_max) and inverse_max > 0.0:
        inverse /= inverse_max

    support_mass = affinity.sum(axis=1)
    normalized_support = affinity / np.maximum(
        support_mass[:, None],
        eps,
    )
    mutual_support = np.sqrt(
        np.maximum(
            normalized_support * normalized_support.T,
            0.0,
        )
    )
    mutual_support[~valid] = 0.0

    rank_order = np.argsort(
        np.where(valid, distances, np.inf),
        axis=1,
        kind="stable",
    )
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], rank_order] = np.arange(n)

    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_scale)
    rank_score[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )

    local_signal = (
        0.65 * mutual_support
        + 0.35 * mutual_rank
    )

    distance_signal = (
        np.maximum(inverse, eps) ** 0.68
        * np.maximum(affinity, eps) ** 0.32
    )

    score = distance_signal * (
        0.90 + 0.10 * local_signal
    )

    valid_floor = 0.025 * np.maximum(
        float(score[valid].max()),
        eps,
    )
    score[valid] = score[valid] + valid_floor
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
