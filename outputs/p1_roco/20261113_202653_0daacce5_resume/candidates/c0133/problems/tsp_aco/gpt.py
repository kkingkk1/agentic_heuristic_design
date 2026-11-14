import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diversified edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    clean = np.where(finite, np.maximum(distances, 0.0), np.inf)
    np.fill_diagonal(clean, np.inf)

    reciprocal = valid & valid.T
    symmetric = bool(
        np.all(reciprocal == valid)
        and np.allclose(
            clean[reciprocal],
            clean.T[reciprocal],
            rtol=1e-8,
            atol=1e-12,
        )
    )

    if symmetric:
        clean = np.where(
            reciprocal,
            0.5 * (clean + clean.T),
            clean,
        )
        valid = reciprocal.copy()
        np.fill_diagonal(valid, False)

    finite_values = clean[valid]
    positive = finite_values[finite_values > 0.0]

    if positive.size:
        center = float(np.median(positive))
        mad = float(np.median(np.abs(positive - center)))
        scale = max(center, 1.4826 * mad, 1.0)
    elif finite_values.size:
        scale = max(float(np.median(finite_values)), 1.0)
    else:
        scale = 1.0

    eps = np.sqrt(np.finfo(float).eps) * scale

    # Blend temperature-scaled and inverse-distance attraction.
    ratio = np.zeros((n, n), dtype=float)
    ratio[valid] = np.minimum(finite_values / (scale + eps), 50.0)

    exponential = np.exp(-ratio)
    inverse = 1.0 / (1.0 + ratio)
    attraction = 0.55 * exponential + 0.45 * inverse

    # Finite local regret; missing alternatives do not create artificial rewards.
    row_sorted = np.sort(np.where(valid, clean, np.inf), axis=1)
    col_sorted = np.sort(np.where(valid, clean, np.inf), axis=0)

    row_best = row_sorted[:, 0]
    col_best = col_sorted[0, :]

    row_second = row_sorted[:, 1] if n > 1 else row_best
    col_second = col_sorted[1, :] if n > 1 else col_best

    row_gap = np.where(
        np.isfinite(row_second),
        np.maximum(row_second - row_best, 0.0),
        0.0,
    )
    col_gap = np.where(
        np.isfinite(col_second),
        np.maximum(col_second - col_best, 0.0),
        0.0,
    )

    row_regret = np.clip(row_gap / (scale + eps), 0.0, 12.0)
    col_regret = np.clip(col_gap / (scale + eps), 0.0, 12.0)

    competition_bonus = (
        1.0
        + 0.16 * np.tanh(row_regret)[:, None]
        + 0.16 * np.tanh(col_regret)[None, :]
    )

    # Approximate bypass savings using a bounded set of promising intermediates.
    candidate_count = min(
        max(2, int(np.sqrt(max(n, 2))) + 2),
        max(n - 1, 1),
    )

    two_hop = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        outgoing = np.flatnonzero(valid[i])
        if outgoing.size == 0:
            continue

        order = outgoing[np.argsort(clean[i, outgoing])]
        intermediates = order[:candidate_count]

        for middle in intermediates:
            via = clean[i, middle] + clean[middle]
            two_hop[i] = np.minimum(two_hop[i], via)

    np.fill_diagonal(two_hop, np.inf)

    savings = np.where(
        valid & np.isfinite(two_hop),
        np.maximum(two_hop - clean, 0.0),
        0.0,
    )
    savings = np.clip(savings / (scale + eps), 0.0, 12.0)

    bypass_bonus = 1.0 + 0.18 * np.tanh(savings)

    raw = attraction * competition_bonus * bypass_bonus
    raw[~valid] = 0.0

    # Encourage edges whose endpoints retain broad future options.
    row_mass = raw.sum(axis=1)
    col_mass = raw.sum(axis=0)

    row_mean = (
        float(np.mean(row_mass[row_mass > 0.0]))
        if np.any(row_mass > 0.0)
        else 1.0
    )
    col_mean = (
        float(np.mean(col_mass[col_mass > 0.0]))
        if np.any(col_mass > 0.0)
        else 1.0
    )

    row_potential = np.sqrt(
        np.clip(row_mass / (row_mean + eps), 0.5, 2.0)
    )
    col_potential = np.sqrt(
        np.clip(col_mass / (col_mean + eps), 0.5, 2.0)
    )

    raw *= row_potential[:, None] * col_potential[None, :]
    raw[~valid] = 0.0

    if symmetric:
        raw = 0.5 * (raw + raw.T)
        raw[~valid] = 0.0

    # Light damped Sinkhorn-style correction for endpoint balance.
    corrected = raw.copy()

    for _ in range(4):
        row_sum = corrected.sum(axis=1)
        active_rows = row_sum > 0.0
        if np.any(active_rows):
            target_row = float(np.mean(row_sum[active_rows]))
            row_factor = np.ones(n, dtype=float)
            row_factor[active_rows] = np.sqrt(
                np.clip(
                    target_row / (row_sum[active_rows] + eps),
                    0.55,
                    1.8,
                )
            )
            corrected *= row_factor[:, None]

        col_sum = corrected.sum(axis=0)
        active_cols = col_sum > 0.0
        if np.any(active_cols):
            target_col = float(np.mean(col_sum[active_cols]))
            col_factor = np.ones(n, dtype=float)
            col_factor[active_cols] = np.sqrt(
                np.clip(
                    target_col / (col_sum[active_cols] + eps),
                    0.55,
                    1.8,
                )
            )
            corrected *= col_factor[None, :]

        corrected[~valid] = 0.0

    if symmetric:
        corrected = 0.5 * (corrected + corrected.T)
        corrected[~valid] = 0.0

    # Concave compression preserves viable alternatives for stochastic sampling.
    priors = np.power(np.maximum(corrected, 0.0), 0.82)
    priors[~valid] = 0.0

    maximum = float(np.max(priors))
    if maximum > 0.0 and np.isfinite(maximum):
        priors /= maximum

    priors[~np.isfinite(priors)] = 0.0
    np.fill_diagonal(priors, 0.0)

    if symmetric:
        priors = 0.5 * (priors + priors.T)
        priors[~valid] = 0.0

    return priors
