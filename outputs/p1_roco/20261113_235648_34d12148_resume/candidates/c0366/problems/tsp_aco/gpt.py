import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diverse, long-term-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(raw)
        & np.isfinite(raw.T)
        & (raw >= 0.0)
        & (raw.T >= 0.0)
        & ~diagonal
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    upper = float(np.quantile(observed, 0.90))
    if not np.isfinite(upper):
        upper = scale

    penalty = max(upper, scale) + 8.0 * scale
    clean = np.where(valid, distances, penalty)
    np.fill_diagonal(clean, penalty)

    spread = float(np.median(np.abs(observed - scale)))
    if not np.isfinite(spread):
        spread = 0.0

    temperature = max(0.58 * scale + 0.25 * spread, eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(clean[valid] / temperature, 50.0)
    )

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    rank_signal = np.exp(-ranks / rank_scale)
    rank_signal[~valid] = 0.0

    # Estimate each endpoint's local density from its closest usable neighbors.
    degree = valid.sum(axis=1)
    nearest_count = max(1, min(5, n - 1))

    nearest = np.where(valid, clean, np.inf)
    nearest = np.sort(nearest, axis=1)[:, :nearest_count]
    nearest = np.where(np.isfinite(nearest), nearest, penalty)

    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    # Sparse endpoints receive only a bounded extra preference.
    sparsity = local_scale / (local_scale + scale)
    sparsity = np.clip(sparsity, 0.0, 0.75)
    sparse_pair = np.sqrt(
        sparsity[:, None] * sparsity[None, :]
    )
    sparse_pair[~valid] = 0.0

    # Compute the best and second-best alternatives for every endpoint.
    best = nearest[:, 0]
    if nearest_count >= 2:
        second = nearest[:, 1]
    else:
        second = best + scale

    # Positive regret means that excluding this edge forces a worse local choice.
    endpoint_regret = np.maximum(second[:, None] - clean, 0.0)
    endpoint_regret /= (
        second[:, None] + clean + scale + eps
    )
    endpoint_regret = np.clip(endpoint_regret, 0.0, 1.0)
    endpoint_regret[~valid] = 0.0

    mutual_advantage = np.sqrt(
        endpoint_regret * endpoint_regret.T
    )
    mutual_advantage[~valid] = 0.0

    affinity_norm = affinity / max(float(affinity.max()), eps)
    rank_pair = np.sqrt(
        np.maximum(rank_signal * rank_signal.T, 0.0)
    )

    # Mildly favor edges whose endpoints still have alternative capacity.
    row_mass = affinity.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)
    probabilities = affinity / row_mass[:, None]
    concentration = np.sum(probabilities * probabilities, axis=1)
    flexibility = 1.0 - np.sqrt(np.clip(concentration, 0.0, 1.0))
    flexibility = np.clip(flexibility, 0.05, 1.0)

    reserve_pair = np.sqrt(
        flexibility[:, None] * flexibility[None, :]
    )
    reserve_pair[~valid] = 0.0

    base = (
        0.58 * affinity_norm
        + 0.15 * rank_pair
        + 0.13 * mutual_advantage
        + 0.08 * sparse_pair
        + 0.06 * reserve_pair
    )

    base[~valid] = 0.0
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)

    maximum = float(base.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)
    base /= maximum

    # One damped global correction prevents a few nodes from dominating.
    row_sum = base.sum(axis=1)
    positive_rows = row_sum[row_sum > 0.0]
    target = (
        float(np.median(positive_rows))
        if positive_rows.size
        else 1.0
    )
    target = max(target, eps)

    node_correction = np.sqrt(
        target / np.maximum(row_sum, eps)
    )
    node_correction = np.clip(node_correction, 0.72, 1.35)

    corrected = base * np.sqrt(
        node_correction[:, None] * node_correction[None, :]
    )
    corrected[~valid] = 0.0
    np.fill_diagonal(corrected, 0.0)

    result = 0.78 * base + 0.22 * corrected
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
