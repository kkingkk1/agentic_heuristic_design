import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, competition-aware edge-priority matrix for TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    finite_nonnegative = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~diagonal
    )

    reverse = finite_nonnegative.T
    both = finite_nonnegative & reverse
    forward_only = finite_nonnegative & ~reverse
    backward_only = reverse & ~finite_nonnegative

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both] = 0.5 * (
        raw[both] + raw.T[both]
    )
    distance[forward_only] = raw[forward_only]
    distance[backward_only] = raw.T[backward_only]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return result

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        reference = float(np.median(positive_values))
    else:
        reference = float(np.median(finite_values))

    reference = max(reference, eps)

    transformed = np.zeros((n, n), dtype=float)
    transformed[valid] = np.log1p(
        distance[valid] / reference
    )

    counts = valid.sum(axis=1)
    row_median = np.full(n, reference, dtype=float)

    for i in range(n):
        values = distance[i, valid[i]]
        if values.size:
            positive = values[values > 0.0]
            if positive.size:
                row_median[i] = float(np.median(positive))
            else:
                row_median[i] = float(np.median(values))

    row_median = np.maximum(row_median, eps)

    row_temperature = np.sqrt(
        row_median * reference
    )
    row_temperature = np.maximum(row_temperature, eps)

    logits = np.full((n, n), -np.inf, dtype=float)
    logits[valid] = -(
        transformed[valid]
        * reference
        / np.sqrt(
            row_temperature[:, None][valid]
            * row_temperature[None, :][valid]
        )
    )

    probabilities = np.zeros((n, n), dtype=float)

    for i in range(n):
        mask = valid[i]
        if not np.any(mask):
            continue

        row_logits = logits[i, mask]
        shift = float(np.max(row_logits))
        weights = np.exp(
            np.clip(row_logits - shift, -60.0, 0.0)
        )
        total = float(np.sum(weights))

        if np.isfinite(total) and total > 0.0:
            probabilities[i, mask] = weights / total

    mutual_probability = np.sqrt(
        probabilities * probabilities.T
    )

    row_max = np.max(
        np.where(valid, probabilities, 0.0),
        axis=1,
    )
    concentration = np.clip(
        counts.astype(float) * row_max,
        1.0,
        float(max(n - 1, 1)),
    )

    concentration_factor = np.sqrt(
        concentration[:, None] * concentration[None, :]
    )
    concentration_factor = 1.0 / np.sqrt(
        np.maximum(concentration_factor, 1.0)
    )

    opportunity = np.ones((n, n), dtype=float)

    for i in range(n):
        values = np.sort(distance[i, valid[i]])
        if values.size == 0:
            continue

        best = values[0]
        second = values[1] if values.size > 1 else values[0]
        gap = max(second - best, 0.0)

        local_mask = valid[i]
        relative_gap = gap / np.maximum(
            distance[i, local_mask] + reference,
            eps,
        )

        opportunity[i, local_mask] += 0.14 * np.clip(
            relative_gap,
            0.0,
            1.0,
        )

    opportunity = np.sqrt(
        opportunity * opportunity.T
    )

    distance_floor = np.maximum(
        np.minimum(row_median[:, None], row_median[None, :]),
        eps,
    )
    direct_advantage = np.exp(
        -np.clip(
            distance / distance_floor,
            0.0,
            40.0,
        )
    )
    direct_advantage[~valid] = 0.0

    scores = (
        np.sqrt(mutual_probability)
        * (0.72 + 0.28 * mutual_probability)
        * (0.82 + 0.18 * concentration_factor)
        * opportunity
        * (0.88 + 0.12 * direct_advantage)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
