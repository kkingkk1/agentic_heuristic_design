import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
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
    scale_floor = max(
        float(np.median(positive)) if positive.size else fallback,
        eps,
    )

    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    ordered = np.sort(d, axis=1)
    local_scale = np.maximum(
        np.mean(ordered[:, :k], axis=1),
        scale_floor * 1e-6,
    )

    normalized = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    short_edge = np.exp(-np.clip(normalized, 0.0, 50.0))

    endpoint_fit_i = np.exp(
        -np.clip(d / local_scale[:, None], 0.0, 50.0)
    )
    endpoint_fit_j = np.exp(
        -np.clip(d / local_scale[None, :], 0.0, 50.0)
    )
    endpoint_compatibility = np.sqrt(endpoint_fit_i * endpoint_fit_j)

    row_order = np.argsort(d, axis=1)
    best_index = row_order[:, 0]
    best_distance = d[np.arange(n), best_index]

    second_distance = ordered[:, 1] if n > 2 else ordered[:, 0]

    alternative_i = np.broadcast_to(
        best_distance[:, None],
        (n, n),
    ).copy()
    nearest_mask = (
        np.arange(n)[None, :] == best_index[:, None]
    )
    alternative_i[nearest_mask] = np.broadcast_to(
        second_distance[:, None],
        (n, n),
    )[nearest_mask]

    alternative_j = alternative_i.T

    margin_i = np.maximum(
        alternative_i - d,
        0.0,
    ) / np.maximum(
        alternative_i + d + scale_floor,
        eps,
    )
    margin_j = np.maximum(
        alternative_j - d,
        0.0,
    ) / np.maximum(
        alternative_j + d + scale_floor,
        eps,
    )

    margin_bonus = 1.0 + 0.15 * np.minimum(
        np.clip(margin_i, 0.0, 1.0),
        np.clip(margin_j, 0.0, 1.0),
    )

    score = (
        short_edge
        * endpoint_compatibility
        * margin_bonus
    )

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
