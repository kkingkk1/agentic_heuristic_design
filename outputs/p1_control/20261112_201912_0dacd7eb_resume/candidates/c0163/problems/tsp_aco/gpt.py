import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)
    positive = distances[
        finite_nonnegative & off_diagonal & (distances > 0.0)
    ]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    cleaned = np.where(finite_nonnegative, distances, fallback)
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)
    np.fill_diagonal(cleaned, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    neighborhood = np.partition(cleaned, k - 1, axis=1)
    local_neighbors = neighborhood[:, :k]
    local_scale = np.maximum(
        np.mean(local_neighbors, axis=1),
        eps,
    )
    substitute = np.maximum(neighborhood[:, k - 1], eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = cleaned / (pair_scale + eps)

    closeness = np.exp(
        -np.clip(relative_length, 0.0, 50.0)
    )

    row_order = np.argsort(
        cleaned,
        axis=1,
        kind="mergesort",
    )
    ranks = np.empty_like(row_order, dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, row_order] = np.arange(n, dtype=float)[None, :]

    rank_scale = max(float(n - 1), 1.0)
    normalized_rank = ranks / rank_scale

    mutual_rank = np.exp(
        -1.35 * (normalized_rank + normalized_rank.T)
    )
    rank_agreement = np.exp(
        -0.8 * np.abs(normalized_rank - normalized_rank.T)
    )

    competitiveness = 1.0 / (1.0 + relative_length)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    substitute_scale = np.sqrt(
        substitute[:, None] * substitute[None, :]
    )
    margin = substitute_scale / (
        cleaned + substitute_scale + eps
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_factor = 0.92 + 0.08 * margin

    score = (
        closeness
        * mutual_rank
        * (0.84 + 0.16 * mutual_competitiveness)
        * (0.82 + 0.18 * rank_agreement)
        * margin_factor
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    if values.size:
        cutoff = float(np.percentile(values, 85.0))
        score *= np.where(score >= cutoff, 1.04, 1.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    score = np.clip(score, 0.0, 1.0)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
