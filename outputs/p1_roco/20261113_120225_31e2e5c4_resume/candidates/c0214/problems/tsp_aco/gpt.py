import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a robust, degree-balanced stochastic prior for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    if not np.any(finite):
        return result

    reverse_finite = finite.T
    valid = finite | reverse_finite

    work = np.full((n, n), np.inf, dtype=float)

    both = finite & reverse_finite
    work[both] = 0.5 * (d[both] + d.T[both])
    work[finite & ~reverse_finite] = d[finite & ~reverse_finite]
    work[~finite & reverse_finite] = d.T[~finite & reverse_finite]

    np.fill_diagonal(work, np.inf)

    finite_values = work[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    typical = np.full(n, global_scale, dtype=float)
    spread = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = work[i, valid[i]]
        if values.size == 0:
            continue

        center = float(np.median(values))
        deviation = float(np.median(np.abs(values - center)))

        typical[i] = center
        spread[i] = max(
            1.4826 * deviation,
            0.20 * global_scale,
            eps,
        )

    pair_spread = np.sqrt(
        np.maximum(spread[:, None] * spread[None, :], eps)
    )
    pair_typical = 0.5 * (
        typical[:, None] + typical[None, :]
    )

    deviation = (work - pair_typical) / pair_spread
    deviation = np.clip(
        np.nan_to_num(
            deviation,
            nan=50.0,
            posinf=50.0,
            neginf=-50.0,
        ),
        -50.0,
        50.0,
    )

    local_affinity = 1.0 / (1.0 + np.exp(deviation))

    absolute_affinity = np.exp(
        -np.clip(work / global_scale, 0.0, 50.0)
    )

    affinity = np.sqrt(local_affinity * absolute_affinity)
    affinity[~valid] = 0.0
    affinity = 0.5 * (affinity + affinity.T)

    row_mass = np.sum(affinity, axis=1)
    soft_choice = affinity / np.maximum(row_mass[:, None], eps)
    bidirectional_confidence = np.sqrt(
        soft_choice * soft_choice.T
    )

    balanced = affinity.copy()

    for _ in range(24):
        mass = np.sum(balanced, axis=1)
        scale = 1.0 / np.sqrt(np.maximum(mass, eps))
        balanced *= scale[:, None] * scale[None, :]
        balanced[~valid] = 0.0
        balanced = 0.5 * (balanced + balanced.T)

    scores = balanced * (
        0.70 + 0.30 * bidirectional_confidence
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
