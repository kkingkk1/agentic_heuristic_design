import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, structurally informed edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, distances, np.inf)

    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        center = float(np.median(positive))
        mad = float(np.median(np.abs(positive - center)))
        scale = max(center, 1.4826 * mad, 1.0)
    else:
        finite = safe[valid]
        scale = max(float(np.median(finite)), 1.0) if finite.size else 1.0

    eps = np.sqrt(np.finfo(float).eps) * scale
    normalized = np.minimum(safe / (scale + eps), 50.0)

    attraction = (
        0.68 * np.exp(-normalized)
        + 0.32 / (1.0 + normalized)
    )
    attraction[~valid] = 0.0

    # Bounded endpoint scarcity: preserve edges that are especially important
    # to one or both endpoints without strongly penalizing alternatives.
    row_sorted = np.sort(safe, axis=1)
    col_sorted = np.sort(safe, axis=0)

    row_best = row_sorted[:, 0]
    col_best = col_sorted[0, :]

    row_second = row_sorted[:, 1]
    col_second = col_sorted[1, :]

    row_gap = np.where(
        np.isfinite(row_second),
        np.maximum(row_second - row_best, 0.0) / (scale + eps),
        0.0,
    )
    col_gap = np.where(
        np.isfinite(col_second),
        np.maximum(col_second - col_best, 0.0) / (scale + eps),
        0.0,
    )

    row_regret = np.minimum(row_gap, 8.0)
    col_regret = np.minimum(col_gap, 8.0)

    row_advantage = np.maximum(
        (safe - row_best[:, None]) / (scale + eps),
        0.0,
    )
    col_advantage = np.maximum(
        (safe - col_best[None, :]) / (scale + eps),
        0.0,
    )

    scarcity = 1.0 + 0.14 * np.tanh(
        np.minimum(row_advantage, 6.0)
        + np.minimum(col_advantage, 6.0)
    )

    # Correct the direction: an edge close to an endpoint's best edge should
    # retain high priority, while a large local regret should modestly favor
    # the endpoint's strongest available choices.
    endpoint_competition = (
        1.0
        + 0.10 * np.tanh(row_regret)[:, None]
        + 0.10 * np.tanh(col_regret)[None, :]
    )

    scores = attraction * scarcity * endpoint_competition
    scores[~valid] = 0.0

    # Restricted two-hop bypass signal. It is deliberately weak and bounded,
    # so direct distance remains the dominant criterion.
    if n >= 3:
        candidate_count = min(3, n - 1)
        two_hop = np.full((n, n), np.inf, dtype=float)

        for i in range(n):
            candidates = np.flatnonzero(valid[i])
            if candidates.size == 0:
                continue

            order = candidates[np.argsort(safe[i, candidates])]
            for middle in order[:candidate_count]:
                via = safe[i, middle] + safe[middle]
                two_hop[i] = np.minimum(two_hop[i], via)

        np.fill_diagonal(two_hop, np.inf)

        usable = valid & np.isfinite(two_hop)
        bypass_signal = np.zeros((n, n), dtype=float)
        bypass_signal[usable] = np.tanh(
            np.clip(
                (two_hop[usable] - safe[usable]) / (scale + eps),
                -6.0,
                6.0,
            )
        )

        scores *= 1.0 + 0.08 * bypass_signal
        scores[~valid] = 0.0

    # If the matrix is effectively symmetric, preserve that property exactly.
    reciprocal = valid & valid.T
    symmetric = (
        np.all(valid == valid.T)
        and np.allclose(
            distances[reciprocal],
            distances.T[reciprocal],
            rtol=1e-8,
            atol=1e-12,
        )
    )

    if symmetric:
        scores = 0.5 * (scores + scores.T)
        scores[~valid] = 0.0

    # Light balancing limits excessive concentration on a single endpoint.
    balanced = scores.copy()
    for _ in range(2):
        row_sum = balanced.sum(axis=1)
        active_rows = row_sum > 0.0
        if np.any(active_rows):
            target = float(np.mean(row_sum[active_rows]))
            factor = np.ones(n, dtype=float)
            factor[active_rows] = np.sqrt(
                np.clip(target / (row_sum[active_rows] + eps), 0.72, 1.40)
            )
            balanced *= factor[:, None]

        col_sum = balanced.sum(axis=0)
        active_cols = col_sum > 0.0
        if np.any(active_cols):
            target = float(np.mean(col_sum[active_cols]))
            factor = np.ones(n, dtype=float)
            factor[active_cols] = np.sqrt(
                np.clip(target / (col_sum[active_cols] + eps), 0.72, 1.40)
            )
            balanced *= factor[None, :]

        balanced[~valid] = 0.0

    scores = 0.84 * scores + 0.16 * balanced
    scores[~valid] = 0.0

    if symmetric:
        scores = 0.5 * (scores + scores.T)
        scores[~valid] = 0.0

    # Concave compression preserves viable alternatives for stochastic sampling.
    priors = np.power(np.maximum(scores, 0.0), 0.86)
    priors[~valid] = 0.0
    priors = np.nan_to_num(priors, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(priors))
    if maximum > 0.0:
        priors /= maximum

    np.fill_diagonal(priors, 0.0)

    if symmetric:
        priors = 0.5 * (priors + priors.T)
        priors[~valid] = 0.0

    return priors
