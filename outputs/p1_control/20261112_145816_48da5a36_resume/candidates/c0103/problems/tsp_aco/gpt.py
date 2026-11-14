import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0)
    valid &= valid.T
    np.fill_diagonal(valid, False)

    positive_values = d[valid]
    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive_values)), eps)
    fallback = max(float(np.max(positive_values)) * 10.0, global_scale)

    safe_d = np.where(valid, d, fallback)
    np.fill_diagonal(safe_d, fallback)

    finite_rows = np.where(valid, d, np.inf)
    sorted_d = np.sort(finite_rows, axis=1)

    # Use a moderately local scale, falling back safely for sparse rows.
    kth = min(2, n - 2)
    local_scale = sorted_d[:, kth]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = safe_d / np.maximum(geometric_scale, eps)

    # Stable mutual rank preference.
    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks /= max(n - 1, 1)
    ranks = np.where(valid, ranks, 1.0)
    mutual_rank = ranks + ranks.T

    # Endpoint compatibility favors edges short for both endpoints.
    endpoint_attractiveness = np.exp(
        -np.minimum(relative_length, 50.0)
    )

    # Penalize edges whose endpoints have very different local scales.
    scale_balance = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )

    # Mild density penalty discourages unusually long edges in dense regions.
    neighborhood = valid & (
        safe_d <= 1.5 * local_scale[:, None]
    )
    density = neighborhood.mean(axis=1)
    density_penalty = (
        density[:, None] + density[None, :]
    ) * np.minimum(relative_length, 2.0)

    log_score = (
        0.85 * np.log(np.maximum(endpoint_attractiveness, eps))
        - 1.35 * mutual_rank
        - 0.35 * safe_d / global_scale
        - 0.22 * scale_balance
        - 0.18 * density_penalty
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
