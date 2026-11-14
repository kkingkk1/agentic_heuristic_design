import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite))
    d = np.where(np.isfinite(d), d, fallback)

    positive = finite[finite > 0.0]
    eps = np.finfo(float).eps
    reference_scale = (
        float(np.median(positive))
        if positive.size
        else max(fallback, eps)
    )
    scale_floor = max(reference_scale * 1e-6, eps)

    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    ordered = np.sort(d, axis=1)

    local_scale = np.maximum(
        np.mean(ordered[:, :k], axis=1),
        scale_floor,
    )

    endpoint_ratio_i = d / local_scale[:, None]
    endpoint_ratio_j = d / local_scale[None, :]

    normalized_distance = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    log_score = -(
        0.50 * np.clip(normalized_distance, 0.0, 50.0)
        + 0.25 * np.clip(endpoint_ratio_i, 0.0, 50.0)
        + 0.25 * np.clip(endpoint_ratio_j, 0.0, 50.0)
    )

    best_distance = ordered[:, 0]
    if n > 2:
        alternative_distance = ordered[:, 1]
    else:
        alternative_distance = best_distance.copy()

    best_index = np.argmin(d, axis=1)
    endpoint_alternative = np.broadcast_to(
        alternative_distance[:, None],
        (n, n),
    ).copy()

    nearest_edges = (
        np.arange(n)[None, :] == best_index[:, None]
    )
    endpoint_alternative[nearest_edges] = alternative_distance[
        np.arange(n)
    ][:, None][nearest_edges]

    alternative_transpose = endpoint_alternative.T

    margin_i = np.maximum(endpoint_alternative - d, 0.0) / np.maximum(
        endpoint_alternative + d + scale_floor,
        eps,
    )
    margin_j = np.maximum(alternative_transpose - d, 0.0) / np.maximum(
        alternative_transpose + d + scale_floor,
        eps,
    )

    distinctiveness = np.minimum(
        np.clip(margin_i, 0.0, 1.0),
        np.clip(margin_j, 0.0, 1.0),
    )
    log_score += np.log1p(0.15 * distinctiveness)

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
