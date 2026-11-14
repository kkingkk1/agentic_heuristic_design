import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge priors for stochastic TSP sampling."""
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

    values = distances[valid]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    finite_values = values[np.isfinite(values)]
    maximum = float(np.max(finite_values)) if finite_values.size else scale
    penalty = maximum + 16.0 * scale

    safe_distances = np.where(valid, distances, penalty)
    np.fill_diagonal(safe_distances, penalty)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(safe_distances[valid] / scale, 40.0)
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe_distances[valid] + eps)
    inverse /= max(float(np.max(inverse)), eps)

    row_mass = affinity.sum(axis=1)
    col_mass = affinity.sum(axis=0)
    row_share = affinity / np.maximum(row_mass[:, None], eps)
    col_share = affinity / np.maximum(col_mass[None, :], eps)
    mutual_support = np.sqrt(np.maximum(row_share * col_share, 0.0))

    density_reference = float(np.median(row_mass[row_mass > 0.0]))
    density_reference = max(density_reference, eps)
    scarcity = density_reference / np.maximum(row_mass, eps)
    scarcity = np.clip(scarcity, 0.35, 3.0)
    endpoint_support = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], eps)
    )
    endpoint_support = np.clip(endpoint_support / 2.0, 0.2, 2.0)

    row_order = np.argsort(safe_distances, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    rank_score = np.exp(
        -row_rank / max(np.sqrt(float(n)), 1.0)
    )
    rank_score[~valid] = 0.0

    sorted_distances = np.sort(safe_distances, axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, min(1, n - 1)]
    regret_scale = float(np.median(np.maximum(second - best, 0.0)))
    regret_scale = max(regret_scale, 0.05 * scale, eps)

    row_regret = np.maximum(second[:, None] - safe_distances, 0.0)
    col_sorted = np.sort(safe_distances.T, axis=1)
    col_second = col_sorted[:, min(1, n - 1)]
    col_regret = np.maximum(col_second[None, :] - safe_distances, 0.0)

    exclusivity = 1.0 - np.exp(
        -0.5 * (row_regret + col_regret) / regret_scale
    )
    exclusivity[~valid] = 0.0

    bypass = np.full((n, n), penalty, dtype=float)
    for middle in range(n):
        candidate = (
            safe_distances[:, middle, None]
            + safe_distances[None, middle, :]
        )
        bypass = np.minimum(bypass, candidate)

    bypass[diagonal] = penalty
    advantage = np.maximum(bypass - safe_distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -advantage / max(scale, eps)
    )
    irreplaceability[~valid] = 0.0

    components = [
        np.maximum(inverse, eps),
        np.maximum(affinity, eps),
        np.maximum(rank_score, eps),
        np.maximum(mutual_support, eps),
        np.maximum(endpoint_support, eps),
        np.maximum(exclusivity, eps),
        np.maximum(irreplaceability, eps),
    ]

    weights = (
        0.30 * np.log(components[0])
        + 0.20 * np.log(components[1])
        + 0.12 * np.log(components[2])
        + 0.12 * np.log(components[3])
        + 0.10 * np.log(components[4])
        + 0.08 * np.log(components[5])
        + 0.08 * np.log(components[6])
    )

    finite_weights = weights[valid]
    center = float(np.median(finite_weights))
    spread = float(np.median(np.abs(finite_weights - center)))
    temperature = float(np.clip(1.0 + spread, 0.85, 1.35))

    prior = np.zeros((n, n), dtype=float)
    prior[valid] = np.exp(
        np.clip((weights[valid] - center) / temperature, -40.0, 0.0)
    )

    exploration = float(np.clip(0.04 + 0.02 * spread, 0.04, 0.08))
    valid_count = float(np.sum(valid))
    prior = (
        (1.0 - exploration) * prior
        + exploration * valid.astype(float) / max(valid_count, 1.0)
    )

    prior = 0.5 * (prior + prior.T)
    prior[~valid] = 0.0

    balanced = prior.copy()
    for _ in range(10):
        row_sum = balanced.sum(axis=1, keepdims=True)
        active = row_sum[:, 0] > eps
        balanced[active] /= row_sum[active]
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    result = 0.78 * prior + 0.22 * balanced
    result = 0.5 * (result + result.T)
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
