import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric entropy-balanced edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if np.any(np.isnan(d)):
        raise ValueError("distance_matrix must not contain NaN values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain nonnegative values")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_offdiag = np.isfinite(d)
    np.fill_diagonal(finite_offdiag, False)

    finite_values = d[finite_offdiag]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    fallback = max(fallback, np.finfo(float).eps)

    safe = d.copy()
    safe[~np.isfinite(safe)] = fallback * 2.0
    safe = 0.5 * (safe + safe.T)
    np.fill_diagonal(safe, 0.0)

    offdiag = ~np.eye(n, dtype=bool)

    temperatures = np.zeros(n, dtype=float)
    affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        values = safe[i, offdiag[i]]
        center = float(np.median(values))
        spread = float(np.median(np.abs(values - center)))
        temperatures[i] = max(
            center + spread,
            np.finfo(float).eps
        )

        logits = -(safe[i] / temperatures[i])
        logits[i] = -np.inf
        row_max = np.max(logits)

        if np.isfinite(row_max):
            weights = np.exp(np.clip(logits - row_max, -745.0, 0.0))
            weights[i] = 0.0
            total = np.sum(weights)
            if total > 0.0 and np.isfinite(total):
                affinity[i] = weights / total

    entropy = np.zeros(n, dtype=float)
    max_entropy = np.log(max(n - 1, 1))

    for i in range(n):
        probabilities = affinity[i]
        positive = probabilities > 0.0
        if np.any(positive):
            entropy[i] = -np.sum(
                probabilities[positive] * np.log(probabilities[positive])
            )

    concentration = 1.0 - entropy / max(max_entropy, np.finfo(float).eps)
    concentration = np.clip(concentration, 0.0, 1.0)

    score = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            direct_affinity = np.sqrt(
                affinity[i, j] * affinity[j, i]
            )

            p = affinity[i].copy()
            q = affinity[j].copy()
            midpoint = 0.5 * (p + q)

            p_positive = p > 0.0
            q_positive = q > 0.0
            midpoint_positive = midpoint > 0.0

            kl_p = 0.0
            kl_q = 0.0

            if np.any(p_positive & midpoint_positive):
                mask = p_positive & midpoint_positive
                kl_p = np.sum(
                    p[mask] * np.log(p[mask] / midpoint[mask])
                )

            if np.any(q_positive & midpoint_positive):
                mask = q_positive & midpoint_positive
                kl_q = np.sum(
                    q[mask] * np.log(q[mask] / midpoint[mask])
                )

            js_divergence = 0.5 * (kl_p + kl_q)
            contextual_compatibility = np.exp(
                -min(max(js_divergence, 0.0), 50.0)
            )

            endpoint_pressure = 1.0 + 0.5 * (
                concentration[i] + concentration[j]
            )

            value = (
                direct_affinity
                * contextual_compatibility
                * endpoint_pressure
            )

            score[i, j] = value
            score[j, i] = value

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
