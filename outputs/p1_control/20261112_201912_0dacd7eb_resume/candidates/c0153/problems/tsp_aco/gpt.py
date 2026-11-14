import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-based edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances)

    positive = distances[finite & (distances > 0.0) & ~diagonal]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    off_diagonal = ~diagonal
    valid_values = d[off_diagonal]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        temperature = float(np.median(positive_values))
        global_scale = float(np.percentile(positive_values, 75.0))
    else:
        temperature = fallback
        global_scale = fallback

    eps = np.finfo(float).eps
    temperature = max(temperature, eps)
    global_scale = max(global_scale, temperature)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    # Adaptive radial kernel. Each row receives its own temperature so that
    # dense and sparse regions contribute comparable transition probabilities.
    finite_rows = np.where(np.isfinite(work), work, fallback)
    row_temperature = np.median(
        np.partition(finite_rows, min(n - 2, max(0, n // 2)), axis=1),
        axis=1,
    )
    row_temperature = np.maximum(
        0.5 * row_temperature + 0.5 * temperature,
        eps,
    )

    kernel = np.exp(
        -np.clip(
            work / row_temperature[:, None],
            0.0,
            50.0,
        )
    )
    np.fill_diagonal(kernel, 0.0)

    row_sum = np.sum(kernel, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, eps)
    transition = kernel / row_sum

    # Two-step diffusion measures whether endpoints share strong intermediate
    # transitions, providing a graph-structural signal distinct from edge rank.
    two_step = transition @ transition
    two_step = np.maximum(two_step, 0.0)

    direct_affinity = np.sqrt(
        np.maximum(transition, 0.0)
        * np.maximum(transition.T, 0.0)
    )

    closure_affinity = np.sqrt(
        np.maximum(two_step, 0.0)
        * np.maximum(two_step.T, 0.0)
    )

    # Prefer edges whose endpoints have balanced transition distributions.
    entropy = -np.sum(
        transition * np.log(np.maximum(transition, eps)),
        axis=1,
    )
    entropy /= max(np.log(max(n - 1, 2)), eps)

    entropy_balance = np.sqrt(
        np.maximum(entropy[:, None], 0.0)
        * np.maximum(entropy[None, :], 0.0)
    )
    entropy_balance = 0.55 + 0.45 * entropy_balance

    # Retain only a weak direct distance term so diffusion structure dominates.
    global_distance = np.exp(
        -np.clip((d / global_scale) ** 0.65, 0.0, 40.0)
    )
    global_distance = 0.70 + 0.30 * global_distance

    score = (
        0.35 * direct_affinity
        + 0.65 * closure_affinity
    )
    score *= entropy_balance
    score *= global_distance

    # Remove self-loops and all numerical artifacts.
    score = 0.5 * (score + score.T)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    if values.size:
        maximum = float(np.max(values))
        if np.isfinite(maximum) and maximum > 0.0:
            score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
