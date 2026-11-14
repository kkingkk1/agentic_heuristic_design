import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    positive_values = d[finite_positive]
    fallback = float(np.max(positive_values)) if positive_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    clean = np.array(d, dtype=float, copy=True)
    clean[~np.isfinite(clean)] = fallback
    clean[clean <= 0] = fallback
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, np.inf)

    eps = np.finfo(float).eps
    valid = np.isfinite(clean) & (clean > 0)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    ordered = np.sort(clean, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = clean / (pair_scale + eps)
    distance_term = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    ranks = np.argsort(np.argsort(clean, axis=1), axis=1).astype(float)
    rank_fraction = ranks / max(float(n - 1), 1.0)
    mutual_rank = 1.0 / (
        1.0 + rank_fraction + rank_fraction.T
    )

    nearest = ordered[:, 0]
    second_nearest = ordered[:, 1] if n > 2 else nearest
    regret = np.maximum(second_nearest - nearest, 0.0)
    positive_regret = regret[regret > 0]
    regret_scale = (
        float(np.median(positive_regret))
        if positive_regret.size
        else float(np.median(local_scale))
    )
    regret_scale = max(regret_scale, eps)

    endpoint_regret = np.sqrt(
        (1.0 + regret[:, None] / regret_scale)
        * (1.0 + regret[None, :] / regret_scale)
    )
    regret_term = 1.0 - np.exp(
        -np.clip(endpoint_regret - 1.0, 0.0, 50.0)
    )

    score = (
        distance_term
        * mutual_rank
        * (0.75 + 0.25 * regret_term)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
