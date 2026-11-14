import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, degree-balanced prior for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    finite = d[valid]
    positive = finite[finite > 0.0]
    base_scale = float(np.median(positive if positive.size else finite))
    if not np.isfinite(base_scale) or base_scale <= 0.0:
        base_scale = 1.0

    forward = np.where(valid, d, np.nan)
    backward = forward.T
    both = np.isfinite(forward) & np.isfinite(backward)

    safe = np.where(
        both,
        0.5 * (forward + backward),
        np.where(np.isfinite(forward), forward, backward),
    )
    edge_valid = np.isfinite(safe)
    np.fill_diagonal(edge_valid, False)
    safe = np.where(edge_valid, safe, np.inf)

    if not np.any(edge_valid):
        return result

    row_values = np.where(edge_valid, safe, np.inf)
    first = np.min(row_values, axis=1)

    if n > 2:
        second = np.partition(row_values, 1, axis=1)[:, 1]
    else:
        second = first.copy()

    finite_first = first[np.isfinite(first)]
    finite_second = second[np.isfinite(second)]

    reference = np.concatenate(
        [finite_first, finite_second]
    ) if finite_second.size else finite_first

    center = float(np.median(reference)) if reference.size else base_scale
    mad = float(np.median(np.abs(reference - center))) if reference.size else 0.0
    temperature = max(1.4826 * mad, 0.15 * base_scale, np.finfo(float).eps)

    safe_first = np.where(np.isfinite(first), first, center)
    safe_second = np.where(np.isfinite(second), second, safe_first)

    regret = np.maximum(safe_second[:, None] - safe, 0.0)
    regret = np.minimum(regret, 8.0 * temperature)

    with np.errstate(over="ignore", invalid="ignore"):
        distance_weight = np.exp(-np.clip(safe / temperature, 0.0, 40.0))
        regret_weight = np.exp(
            np.clip(
                0.5 * (regret + regret.T) / temperature,
                0.0,
                8.0,
            )
        )

    base = distance_weight * regret_weight
    base[~edge_valid] = 0.0
    base = 0.5 * (base + base.T)

    # Symmetric matrix balancing approximates a degree-constrained
    # fractional 2-factor without introducing a separate candidate set.
    scale = np.ones(n, dtype=float)
    for _ in range(32):
        balanced = base * scale[:, None] * scale[None, :]
        row_sum = balanced.sum(axis=1)
        update = np.where(row_sum > 0.0, 1.0 / np.sqrt(row_sum), 1.0)
        scale *= np.clip(update, 0.25, 4.0)
        scale /= max(float(np.max(scale)), 1.0)

    score = base * scale[:, None] * scale[None, :]

    degree = score.sum(axis=1)
    saturation = 1.0 / np.sqrt(np.maximum(degree[:, None] * degree[None, :], 1e-12))
    score *= 0.75 + 0.25 * np.clip(saturation, 0.0, 1.0)

    score[~edge_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
