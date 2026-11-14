import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized edge-priority matrix for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    positive_values = distances[finite_positive]

    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive_values, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    valid = np.isfinite(distances) & (distances >= 0.0)
    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)

    d = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    positive_scale = float(np.median(positive_values))
    scale_floor = max(positive_scale * 1e-10, eps)

    k = max(1, min(n - 1, int(np.rint(np.sqrt(n)))))

    ordered = np.sort(work, axis=1)
    nearest = ordered[:, :k]

    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        scale_floor,
    )

    alternative = np.maximum(
        ordered[:, k - 1],
        scale_floor,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / (pair_scale + scale_floor)
    distance_attraction = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )

    relative_i = d / (local_scale[:, None] + scale_floor)
    relative_j = d / (local_scale[None, :] + scale_floor)

    endpoint_i = np.exp(-np.clip(relative_i, 0.0, 40.0))
    endpoint_j = np.exp(-np.clip(relative_j, 0.0, 40.0))

    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    competition_i = 1.0 / (1.0 + relative_i)
    competition_j = 1.0 / (1.0 + relative_j)
    competition = np.sqrt(competition_i * competition_j)
    competition_factor = 0.84 + 0.16 * competition

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))
    reciprocal_rank = np.exp(
        -1.15
        * np.sqrt(
            np.clip(
                (ranks / max(float(n - 1), 1.0))
                * (ranks.T / max(float(n - 1), 1.0)),
                0.0,
                1.0,
            )
        )
    )

    candidate_support = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )
    candidate_gate = 0.84 + 0.16 * candidate_support

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + scale_floor
    )
    margin_factor = 0.95 + 0.05 * np.clip(margin, 0.0, 1.0)

    local_affinity = np.exp(
        -np.clip(work / (local_scale[:, None] + scale_floor), 0.0, 40.0)
    )
    np.fill_diagonal(local_affinity, 0.0)

    support = local_affinity @ local_affinity.T
    support_norm = np.sqrt(
        np.sum(local_affinity * local_affinity, axis=1)[:, None]
        * np.sum(local_affinity * local_affinity, axis=1)[None, :]
    )
    normalized_support = np.divide(
        support,
        support_norm + scale_floor,
        out=np.zeros_like(support),
        where=np.isfinite(support_norm),
    )
    normalized_support = np.clip(normalized_support, 0.0, 1.0)
    support_factor = 0.97 + 0.03 * normalized_support

    score = (
        distance_attraction
        * endpoint_compatibility
        * reciprocal_rank
        * candidate_gate
        * competition_factor
        * margin_factor
        * support_factor
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
