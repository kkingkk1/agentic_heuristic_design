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

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(1.0, scale)
    upper = float(np.quantile(valid_values, 0.95))
    penalty = max(scale, upper) + 12.0 * scale

    clean = np.where(valid, distances, penalty)
    np.fill_diagonal(clean, penalty)

    temperature = max(0.68 * scale, eps)
    scaled = np.minimum(clean / temperature, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-scaled[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (clean[valid] + eps)
    inverse_max = float(inverse.max())
    if inverse_max > 0.0 and np.isfinite(inverse_max):
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

    rank_scale = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_scale)
    rank_score[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )

    scarcity = 1.0 / np.sqrt(np.maximum(support_mass, eps))
    scarcity_mean = float(np.mean(scarcity))
    if scarcity_mean > 0.0 and np.isfinite(scarcity_mean):
        scarcity /= scarcity_mean
    scarcity = np.clip(scarcity, 0.65, 1.8)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )
    endpoint_scarcity[~valid] = 0.0

    nearest = np.min(clean, axis=1)
    second = np.partition(clean, 1, axis=1)[:, 1]
    regret_scale = np.maximum(
        second - nearest,
        0.18 * scale,
    )

    endpoint_regret = np.maximum(
        second[:, None] - clean,
        0.0,
    ) / regret_scale[:, None]
    endpoint_regret = 1.0 - np.exp(
        -np.minimum(endpoint_regret, 24.0)
    )
    endpoint_regret[~valid] = 0.0

    mutual_regret = np.sqrt(
        np.maximum(
            endpoint_regret * endpoint_regret.T,
            0.0,
        )
    )

    candidate_count = min(
        n - 1,
        max(6, int(np.ceil(np.sqrt(float(n))))),
    )
    nearest_candidates = np.argsort(
        clean,
        axis=1,
        kind="stable",
    )[:, :candidate_count]

    bypass = np.full((n, n), penalty, dtype=float)
    block_size = max(
        1,
        min(
            n,
            1_500_000 // max(n * candidate_count, 1),
        ),
    )

    columns = np.arange(n)[None, None, :]

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        rows = np.arange(start, stop)
        candidates = nearest_candidates[rows]

        first_leg = clean[rows[:, None], candidates]
        second_leg = clean[candidates, :]

        via = first_leg[:, :, None] + second_leg[None, :, :]
        endpoint_mask = candidates[:, :, None] == columns
        via = np.where(endpoint_mask, np.inf, via)

        local_bypass = np.min(via, axis=1)
        bypass[rows] = np.minimum(bypass[rows], local_bypass)

    bypass = 0.5 * (bypass + bypass.T)
    np.fill_diagonal(bypass, penalty)

    advantage = np.maximum(bypass - clean, 0.0)
    irreplaceability = 1.0 - np.exp(
        -advantage / max(0.72 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    local_signal = (
        0.38 * mutual_support
        + 0.27 * mutual_rank
        + 0.35 * mutual_regret
    )

    structural_signal = (
        0.62 * irreplaceability
        + 0.23 * endpoint_scarcity
        + 0.15 * mutual_regret
    )

    score = (
        np.maximum(inverse, eps) ** 0.72
        * np.maximum(affinity, eps) ** 0.78
        * np.maximum(0.30 + local_signal, eps)
        * np.maximum(0.52 + structural_signal, eps)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    score /= maximum

    balanced = score.copy()
    for _ in range(20):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    result = 0.64 * score + 0.36 * balanced
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
