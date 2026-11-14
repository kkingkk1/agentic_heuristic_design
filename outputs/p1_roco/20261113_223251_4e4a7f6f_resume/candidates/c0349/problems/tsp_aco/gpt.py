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
    valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & np.isfinite(raw.T)
        & (raw.T >= 0.0)
        & ~diagonal
    )

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
    upper = float(np.quantile(values, 0.95))
    penalty = max(upper, scale) + 10.0 * scale

    clean = np.where(valid, distances, penalty)
    np.fill_diagonal(clean, penalty)

    temperature = max(0.64 * scale, eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(clean[valid] / temperature, 60.0)
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (clean[valid] + eps)
    inverse /= max(float(inverse.max()), eps)

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

    rank_scale = max(1.0, 0.72 * np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_scale)
    rank_score[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(
            rank_score * rank_score.T,
            0.0,
        )
    )

    scarcity = 1.0 / np.sqrt(np.maximum(support_mass, eps))
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.72, 1.65)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )
    endpoint_scarcity[~valid] = 0.0

    nearest = np.min(clean, axis=1)
    second = np.partition(clean, 1, axis=1)[:, 1]
    gap_scale = np.maximum(
        second - nearest,
        0.14 * scale,
    )

    endpoint_gap = np.where(
        clean <= nearest[:, None] + eps,
        second[:, None] - clean,
        nearest[:, None] - clean,
    )

    exclusivity = np.exp(
        np.clip(
            endpoint_gap / gap_scale[:, None],
            -24.0,
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
    block_size = max(1, min(n, 2_000_000 // max(n, 1)))

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
    irreplaceability = 1.0 - np.exp(
        -advantage / max(0.70 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    local_signal = (
        0.50 * mutual_support
        + 0.32 * mutual_rank
        + 0.18 * mutual_exclusivity
    )

    structural_signal = (
        0.78 * irreplaceability
        + 0.22 * endpoint_scarcity
    )

    score = (
        np.maximum(inverse, eps) ** 0.82
        * np.maximum(affinity, eps) ** 0.78
        * np.maximum(0.32 + local_signal, eps)
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
    for _ in range(36):
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
