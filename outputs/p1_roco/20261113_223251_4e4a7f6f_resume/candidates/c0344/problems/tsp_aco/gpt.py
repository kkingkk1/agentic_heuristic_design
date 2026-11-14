import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    valid_values = distances[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    upper = float(np.quantile(valid_values, 0.90))
    if not np.isfinite(upper):
        upper = scale

    penalty = max(upper, scale) + 16.0 * scale
    clean = np.where(valid, distances, penalty)
    np.fill_diagonal(clean, penalty)

    temperature = max(0.68 * scale, eps)
    normalized_distance = np.minimum(clean / temperature, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-normalized_distance[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(clean[valid], eps)
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

    rank_order = np.argsort(clean, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], rank_order] = np.arange(n)

    rank_temperature = max(1.25, 0.85 * np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_temperature)
    rank_score[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(
            rank_score * rank_score.T,
            0.0,
        )
    )

    valid_degree = valid.sum(axis=1).astype(float)
    scarcity = 1.0 / np.sqrt(np.maximum(valid_degree, 1.0))
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.65, 1.8)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )
    endpoint_scarcity[~valid] = 0.0

    nearest = np.min(clean, axis=1)

    if n >= 2:
        second = np.partition(clean, 1, axis=1)[:, 1]
    else:
        second = nearest.copy()

    local_gap = np.maximum(second - nearest, 0.12 * scale)
    endpoint_gap = np.maximum(second[:, None] - clean, 0.0)

    exclusivity = np.exp(
        -np.minimum(
            endpoint_gap / np.maximum(local_gap[:, None], eps),
            24.0,
        )
    )
    exclusivity[~valid] = 0.0

    mutual_exclusivity = np.sqrt(
        np.maximum(
            exclusivity * exclusivity.T,
            0.0,
        )
    )

    bypass = np.full((n, n), penalty, dtype=float)
    block_size = max(1, min(n, 1_500_000 // max(n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        via = (
            clean[:, start:stop, None]
            + clean[None, start:stop, :]
        )
        bypass = np.minimum(
            bypass,
            np.min(via, axis=1),
        )

    np.fill_diagonal(bypass, penalty)

    advantage = np.maximum(bypass - clean, 0.0)
    protection = 1.0 - np.exp(
        -advantage / max(0.72 * scale, eps)
    )
    protection[~valid] = 0.0

    local_signal = (
        0.48 * mutual_support
        + 0.32 * mutual_rank
        + 0.20 * mutual_exclusivity
    )

    structural_signal = (
        0.74 * protection
        + 0.26 * endpoint_scarcity
    )

    log_score = np.zeros((n, n), dtype=float)

    log_score[valid] = (
        0.86 * np.log(np.maximum(inverse[valid], eps))
        + 0.74 * np.log(np.maximum(affinity[valid], eps))
        + np.log(np.maximum(0.32 + local_signal[valid], eps))
        + np.log(np.maximum(0.58 + structural_signal[valid], eps))
    )

    finite_log_score = log_score[valid]
    if finite_log_score.size == 0 or not np.any(np.isfinite(finite_log_score)):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(finite_log_score))
    score = np.zeros((n, n), dtype=float)
    score[valid] = np.exp(
        np.clip(log_score[valid] - shift, -60.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    score /= maximum

    balanced = score.copy()
    for _ in range(24):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

        row_max = float(balanced.max())
        if np.isfinite(row_max) and row_max > 0.0:
            balanced /= row_max

    result = 0.64 * score + 0.36 * balanced
    result = 0.5 * (result + result.T)
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
