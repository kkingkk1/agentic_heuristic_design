import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized soft edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    valid = finite & nonnegative
    np.fill_diagonal(valid, False)
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (raw + raw.T)
    distance = np.full((n, n), np.inf, dtype=float)
    distance[valid] = np.maximum(symmetric[valid], 0.0)
    np.fill_diagonal(distance, np.inf)

    values = distance[valid]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    scale = max(scale, eps)

    normalized = np.zeros((n, n), dtype=float)
    normalized[valid] = distance[valid] / scale

    # Robust short-edge attraction. The logarithmic form prevents
    # extremely short edges from completely dominating the distribution.
    short = np.zeros((n, n), dtype=float)
    short[valid] = 1.0 / (1.0 + normalized[valid])

    # Exponential affinity is used only for local competition and endpoint
    # statistics, where relative differences are more useful than raw scale.
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-np.minimum(normalized[valid], 30.0))

    row_mass = affinity.sum(axis=1)
    col_mass = affinity.sum(axis=0)

    row_share = np.divide(
        affinity,
        np.maximum(row_mass[:, None], eps),
        out=np.zeros_like(affinity),
        where=row_mass[:, None] > 0.0,
    )
    col_share = np.divide(
        affinity,
        np.maximum(col_mass[None, :], eps),
        out=np.zeros_like(affinity),
        where=col_mass[None, :] > 0.0,
    )

    # Edges that are competitive from both endpoints receive higher prior.
    competition = np.sqrt(np.maximum(row_share * col_share, 0.0))

    # Node scarcity: an edge is more useful when both endpoints have few
    # similarly attractive alternatives.
    row_best = np.max(np.where(valid, affinity, 0.0), axis=1)
    col_best = np.max(np.where(valid, affinity, 0.0), axis=0)

    row_scarcity = np.divide(
        row_best,
        np.maximum(row_mass, eps),
        out=np.zeros_like(row_best),
        where=row_mass > 0.0,
    )
    col_scarcity = np.divide(
        col_best,
        np.maximum(col_mass, eps),
        out=np.zeros_like(col_best),
        where=col_mass > 0.0,
    )

    endpoint = np.sqrt(
        np.maximum(row_scarcity[:, None] * col_scarcity[None, :], 0.0)
    )

    # Local regret: distinguish the best edge from the next-best alternatives
    # at both endpoints.
    large = np.nanmax(values) + 12.0 * scale
    row_sorted = np.sort(np.where(valid, distance, large), axis=1)
    col_sorted = np.sort(np.where(valid, distance.T, large), axis=1)

    row_best_distance = row_sorted[:, 0]
    row_second_distance = row_sorted[:, min(1, n - 1)]
    col_best_distance = col_sorted[:, 0]
    col_second_distance = col_sorted[:, min(1, n - 1)]

    row_regret = np.maximum(
        row_second_distance[:, None] - distance,
        0.0,
    )
    col_regret = np.maximum(
        col_second_distance[None, :] - distance,
        0.0,
    )

    regret_scale = float(
        np.median(
            np.concatenate(
                [
                    np.maximum(row_second_distance - row_best_distance, 0.0),
                    np.maximum(col_second_distance - col_best_distance, 0.0),
                ]
            )
        )
    )
    regret_scale = max(regret_scale, scale * 1e-3)

    regret = 1.0 - np.exp(
        -0.75 * (row_regret + col_regret) / regret_scale
    )
    regret[~valid] = 0.0

    # Two-hop bypass cost estimates how much worse the best indirect route is
    # than using the direct edge.
    bypass = np.full((n, n), large, dtype=float)
    for middle in range(n):
        via_middle = (
            distance[:, middle, None] + distance[None, middle, :]
        )
        bypass = np.minimum(bypass, via_middle)

    bypass[~np.isfinite(bypass)] = large
    np.fill_diagonal(bypass, large)

    bypass_gain = np.maximum(bypass - distance, 0.0)
    irreplaceability = 1.0 - np.exp(
        -0.85 * bypass_gain / scale
    )
    irreplaceability[~valid] = 0.0

    # Normalize structural features into the log domain. Small floors keep
    # valid edges numerically representable without rewarding invalid edges.
    floor = 1e-8

    log_score = (
        0.42 * np.log(np.maximum(short, floor))
        + 0.20 * np.log(np.maximum(competition, floor))
        + 0.10 * np.log(np.maximum(endpoint, floor))
        + 0.16 * np.log(np.maximum(regret, floor))
        + 0.12 * np.log(np.maximum(irreplaceability, floor))
    )

    log_distances = np.log1p(values / scale)
    dispersion = float(
        np.median(np.abs(log_distances - np.median(log_distances)))
    )

    temperature = float(np.clip(0.88 + 0.32 * dispersion, 0.88, 1.28))
    exploration = float(np.clip(0.06 + 0.035 * dispersion, 0.06, 0.105))

    masked = np.where(valid, log_score / temperature, -np.inf)
    maximum = float(np.max(masked[valid]))

    weights = np.zeros((n, n), dtype=float)
    weights[valid] = np.exp(
        np.clip(masked[valid] - maximum, -50.0, 0.0)
    )

    valid_count = float(np.count_nonzero(valid))
    weights = (
        (1.0 - exploration) * weights
        + exploration * valid.astype(float) / max(valid_count, 1.0)
    )

    weights = 0.5 * (weights + weights.T)

    # Alternating normalization reduces row imbalance while preserving
    # symmetry, which is useful for symmetric TSP instances.
    for _ in range(16):
        row_sum = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(row_sum, eps)
        weights = 0.5 * (weights + weights.T)

    direct_prior = np.zeros((n, n), dtype=float)
    direct_prior[valid] = np.sqrt(
        np.maximum(short[valid] * affinity[valid], 0.0)
    )

    result = 0.72 * weights + 0.28 * direct_prior
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
