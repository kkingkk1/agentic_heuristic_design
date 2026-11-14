import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-priority indicators for stochastic TSP solution sampling.
    Larger values indicate more promising directed edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_positive = np.isfinite(d) & (d > 0)
    positive_values = d[finite_positive]

    if positive_values.size == 0:
        return result

    scale = float(np.median(positive_values))
    fallback = max(scale, 1e-12) * 1e6

    safe_d = np.where(
        np.isfinite(d) & (d >= 0),
        d,
        fallback,
    ).astype(float, copy=False)

    np.fill_diagonal(safe_d, np.inf)

    valid = np.isfinite(safe_d) & ~np.eye(n, dtype=bool)
    if not np.any(valid):
        return result

    # Base proximity signal.
    proximity = np.zeros((n, n), dtype=float)
    proximity[valid] = 1.0 / (safe_d[valid] + 1e-12)

    proximity_max = float(np.max(proximity))
    if proximity_max > 0:
        proximity /= proximity_max

    # Soft reciprocal-neighborhood compatibility.
    # A high value means that i considers j good and j also considers i good.
    row_order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(row_order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, row_order[i]] = rank_values

    rank_scale = max(float(n - 1), 1.0)
    reciprocal = np.zeros((n, n), dtype=float)
    reciprocal[valid] = np.exp(
        -(ranks[valid] + ranks.T[valid]) / rank_scale
    )

    reciprocal_max = float(np.max(reciprocal))
    if reciprocal_max > 0:
        reciprocal /= reciprocal_max

    # Robust nearest-alternative baseline for a Clarke-Wright-style saving signal.
    sorted_distances = np.sort(safe_d, axis=1)
    nearest = sorted_distances[:, 0]
    alternative = sorted_distances[:, 1]

    savings = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if valid[i, j]:
                baseline = alternative[i] + alternative[j]
                savings[i, j] = max(0.0, baseline - safe_d[i, j])

    positive_savings = savings[savings > 0]
    if positive_savings.size:
        saving_scale = float(np.percentile(positive_savings, 90))
        saving_scale = max(saving_scale, 1e-12)
        savings = np.minimum(savings / saving_scale, 1.0)

    # Soft mutual-nearest bonus, avoiding a brittle binary gate.
    nearest_index = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_index[i]
        if j != i and nearest_index[j] == i and valid[i, j]:
            mutual[i, j] = 1.0

    # Balanced mixture: savings and reciprocal compatibility are emphasized,
    # while proximity remains the stable fallback signal.
    score = (
        0.32 * proximity
        + 0.30 * reciprocal
        + 0.30 * savings
        + 0.08 * mutual
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    # Temper extreme peaks to preserve sampling diversity.
    score = np.sqrt(np.maximum(score, 0.0))

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
