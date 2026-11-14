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
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distances, np.inf)

    values = distances[valid]
    positive = values[values > 0.0]

    if values.size:
        q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    else:
        q25 = q50 = q75 = 1.0

    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(q50)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    upper = max(
        float(np.quantile(values, 0.95)),
        float(np.quantile(values, 0.75)),
        safe_scale,
    )
    penalty = upper + 16.0 * safe_scale

    clean = np.where(valid, distances, penalty)
    np.fill_diagonal(clean, penalty)

    normalized = np.minimum(clean / safe_scale, 60.0)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-normalized[valid])

    best_distance = np.min(clean, axis=1, keepdims=True)
    temperature = max(0.30 * safe_scale, eps)

    local_soft = np.zeros((n, n), dtype=float)
    local_soft[valid] = np.exp(
        -np.minimum(
            (clean[valid] - best_distance.repeat(n, axis=1)[valid])
            / temperature,
            60.0,
        )
    )

    soft_mass = np.maximum(
        local_soft.sum(axis=1, keepdims=True),
        eps,
    )
    local_probability = local_soft / soft_mass

    mutual_probability = np.sqrt(
        np.maximum(
            local_probability * local_probability.T,
            0.0,
        )
    )
    mutual_probability[~valid] = 0.0

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.80 * np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_scale)
    mutual_rank = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )
    mutual_rank[~valid] = 0.0

    sorted_distances = np.sort(clean, axis=1)
    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, min(1, n - 1)]
    gap = np.maximum(
        second - nearest,
        0.16 * safe_scale,
    )
    gap = np.maximum(gap, eps)

    regret = np.zeros((n, n), dtype=float)
    regret[valid] = np.exp(
        -np.minimum(
            (
                clean[valid]
                - nearest[:, None].repeat(n, axis=1)[valid]
            )
            / gap[:, None].repeat(n, axis=1)[valid],
            60.0,
        )
    )
    mutual_regret = np.sqrt(
        np.maximum(regret * regret.T, 0.0)
    )
    mutual_regret[~valid] = 0.0

    scarcity = 1.0 / np.sqrt(
        np.maximum(soft_mass[:, 0], eps)
    )
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.50, 2.00)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )
    endpoint_scarcity[~valid] = 0.0

    best_two_hop = np.full((n, n), penalty, dtype=float)

    for intermediate in range(n):
        candidate = (
            clean[:, intermediate, None]
            + clean[None, intermediate, :]
        )
        best_two_hop = np.minimum(best_two_hop, candidate)

    for i in range(n):
        best_two_hop[i, i] = penalty

    bypass_gain = np.maximum(
        best_two_hop - clean,
        0.0,
    )
    bypass_scale = max(0.70 * safe_scale, eps)
    irreplaceability = np.zeros((n, n), dtype=float)
    irreplaceability[valid] = 1.0 - np.exp(
        -bypass_gain[valid] / bypass_scale
    )

    affinity_max = float(np.max(affinity))
    if np.isfinite(affinity_max) and affinity_max > 0.0:
        normalized_affinity = affinity / affinity_max
    else:
        normalized_affinity = valid.astype(float)

    score = (
        0.30 * normalized_affinity
        + 0.20 * mutual_probability
        + 0.16 * mutual_rank
        + 0.14 * mutual_regret
        + 0.12 * irreplaceability
        + 0.08 * endpoint_scarcity
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()

    for _ in range(24):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.68 * score + 0.32 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    result = np.maximum(result, 0.0)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
