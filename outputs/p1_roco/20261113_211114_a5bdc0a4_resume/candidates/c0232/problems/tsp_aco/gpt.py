import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized soft edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    distance = 0.5 * (raw + raw.T)
    distance = np.where(valid, np.maximum(distance, 0.0), np.inf)
    np.fill_diagonal(distance, np.inf)

    finite = distance[valid]
    positive = finite[finite > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    finite_log = np.log1p(finite / scale)
    log_center = float(np.median(finite_log))
    log_spread = float(np.median(np.abs(finite_log - log_center)))
    temperature = float(np.clip(0.72 + 0.42 * log_spread, 0.72, 1.24))

    normalized = np.where(valid, distance / scale, np.inf)

    # Primary short-edge preference in a robust bounded form.
    short_edge = np.zeros((n, n), dtype=float)
    short_edge[valid] = np.exp(
        -np.minimum(normalized[valid], 40.0)
    )

    # Soft local competition: an edge is stronger when it is near the
    # best alternatives of both endpoints.
    row_logits = np.where(valid, -normalized, -np.inf)
    row_max = np.max(row_logits, axis=1, keepdims=True)
    row_exp = np.where(valid, np.exp(np.clip(row_logits - row_max, -50.0, 0.0)), 0.0)
    row_prob = row_exp / np.maximum(row_exp.sum(axis=1, keepdims=True), eps)

    col_exp = row_exp.T
    col_prob = col_exp / np.maximum(col_exp.sum(axis=1, keepdims=True), eps)
    col_prob = col_prob.T

    competition = np.sqrt(np.maximum(row_prob * col_prob, eps))

    # Node-level scarcity rewards edges incident to nodes with few
    # attractive alternatives.
    row_soft_degree = np.sum(np.exp(-np.minimum(normalized, 30.0)) * valid, axis=1)
    scarcity = 1.0 / np.sqrt(np.maximum(row_soft_degree, eps))
    scarcity = scarcity / max(float(np.max(scarcity)), eps)
    node_scarcity = np.sqrt(scarcity[:, None] * scarcity[None, :])

    # Relative local gap gives an edge preference when it is much better
    # than the next alternatives at either endpoint.
    row_sorted = np.sort(np.where(valid, normalized, np.inf), axis=1)
    row_best = row_sorted[:, 0]
    row_second = row_sorted[:, min(1, n - 1)]
    row_gap = np.maximum(row_second[:, None] - normalized, 0.0)

    col_sorted = np.sort(np.where(valid, normalized.T, np.inf), axis=1)
    col_second = col_sorted[:, min(1, n - 1)]
    col_gap = np.maximum(col_second[None, :] - normalized, 0.0)

    gap_scale = float(np.median(np.maximum(row_second - row_best, 0.0)))
    gap_scale = max(gap_scale, 1e-3)
    exclusivity = 1.0 - np.exp(
        -0.85 * (row_gap + col_gap) / gap_scale
    )
    exclusivity[~valid] = 0.0

    # Two-hop replacement cost estimates how difficult it is to bypass
    # a direct edge through another node.
    bypass = np.full((n, n), np.inf, dtype=float)
    for middle in range(n):
        left = distance[:, middle][:, None]
        right = distance[middle, :][None, :]
        bypass = np.minimum(bypass, left + right)

    bypass[diagonal] = np.inf
    advantage = np.maximum(bypass - distance, 0.0)
    advantage[~valid] = 0.0

    advantage_scale = float(np.median(finite))
    advantage_scale = max(advantage_scale, scale * 0.25, eps)
    irreplaceability = 1.0 - np.exp(
        -0.85 * advantage / advantage_scale
    )
    irreplaceability[~valid] = 0.0

    def safe_log(values: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(values, eps))

    score = (
        0.44 * safe_log(short_edge)
        + 0.22 * safe_log(competition)
        + 0.12 * safe_log(node_scarcity)
        + 0.11 * safe_log(exclusivity)
        + 0.11 * safe_log(irreplaceability)
    )

    score = np.where(valid, score / temperature, -np.inf)
    maximum = float(np.max(score[valid]))

    weights = np.zeros((n, n), dtype=float)
    weights[valid] = np.exp(np.clip(score[valid] - maximum, -45.0, 0.0))

    dispersion = float(np.std(finite_log))
    exploration = float(np.clip(0.06 + 0.035 * dispersion, 0.06, 0.12))

    degree = np.sum(valid, axis=1, keepdims=True)
    degree_prior = np.where(
        valid,
        1.0 / np.maximum(degree, 1.0),
        0.0,
    )
    degree_prior = 0.5 * (degree_prior + degree_prior.T)

    weights = (
        (1.0 - exploration) * weights
        + exploration * degree_prior
    )
    weights = 0.5 * (weights + weights.T)

    # Symmetric degree balancing prevents a few nodes from monopolizing
    # probability mass while preserving edge ranking.
    for _ in range(18):
        row_sum = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(row_sum, eps)
        weights = 0.5 * (weights + weights.T)

    base = np.sqrt(np.maximum(short_edge * competition, 0.0))
    result = 0.78 * weights + 0.22 * base
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
