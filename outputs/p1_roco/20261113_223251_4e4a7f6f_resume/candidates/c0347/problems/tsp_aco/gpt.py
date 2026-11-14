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

    distance = np.full((n, n), np.inf, dtype=float)
    distance[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distance, np.inf)

    valid_values = distance[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    upper = float(np.quantile(valid_values, 0.95))
    penalty = max(scale, upper) + 16.0 * scale

    clean = np.where(valid, distance, penalty)
    np.fill_diagonal(clean, penalty)

    positive = valid_values[valid_values > 0.0]
    if positive.size:
        lower = float(np.quantile(positive, 0.25))
        upper_scale = float(np.quantile(positive, 0.75))
        spread = max(upper_scale - lower, 0.25 * scale)
    else:
        spread = scale

    temperature = max(0.55 * scale + 0.45 * spread, eps)

    scaled = np.minimum(clean / temperature, 60.0)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-scaled[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(clean[valid], eps)
    inverse_max = float(inverse.max())
    if np.isfinite(inverse_max) and inverse_max > 0.0:
        inverse /= inverse_max

    support = affinity.sum(axis=1)
    support_safe = np.maximum(support, eps)

    endpoint_probability = affinity / support_safe[:, None]
    mutual_probability = np.sqrt(
        np.maximum(endpoint_probability * endpoint_probability.T, 0.0)
    )
    mutual_probability[~valid] = 0.0

    rank_order = np.argsort(clean, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], rank_order] = np.arange(n)

    rank_temperature = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_preference = np.exp(-rank / rank_temperature)
    rank_preference[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_preference * rank_preference.T, 0.0)
    )

    best = np.min(clean, axis=1)
    second = np.partition(clean, 1, axis=1)[:, 1]
    regret_scale = np.maximum(second - best, 0.12 * scale)

    regret = np.zeros((n, n), dtype=float)
    regret[valid] = np.maximum(
        second[:, None] - clean,
        0.0,
    )[valid] / regret_scale[:, None][valid]

    regret = np.clip(regret, 0.0, 12.0)
    regret = 1.0 - np.exp(-regret)
    mutual_regret = np.sqrt(
        np.maximum(regret * regret.T, 0.0)
    )

    scarcity = 1.0 / np.sqrt(support_safe)
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.70, 1.70)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    endpoint_scarcity[~valid] = 0.0

    bypass = np.full((n, n), penalty, dtype=float)
    block_size = max(1, min(n, 2_000_000 // max(n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        via = (
            clean[:, start:stop, None]
            + clean[None, start:stop, :]
        )
        bypass = np.minimum(bypass, np.min(via, axis=1))

    np.fill_diagonal(bypass, penalty)

    replacement_margin = np.maximum(bypass - clean, 0.0)
    irreplaceability = 1.0 - np.exp(
        -replacement_margin / max(0.70 * scale + 0.30 * spread, eps)
    )
    irreplaceability[~valid] = 0.0

    local_signal = (
        0.42 * mutual_probability
        + 0.30 * mutual_rank
        + 0.28 * mutual_regret
    )

    structural_signal = (
        0.62 * irreplaceability
        + 0.23 * endpoint_scarcity
        + 0.15 * mutual_regret
    )

    score = (
        np.maximum(inverse, eps) ** 0.60
        * np.maximum(affinity, eps) ** 0.85
        * np.maximum(0.30 + local_signal, eps) ** 1.10
        * np.maximum(0.50 + structural_signal, eps) ** 0.90
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
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

    result = 0.72 * score + 0.28 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
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
