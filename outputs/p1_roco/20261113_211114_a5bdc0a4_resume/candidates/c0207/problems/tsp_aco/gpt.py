import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return entropy-regularized matching priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    directed_valid = finite & nonnegative
    directed_valid &= ~np.eye(n, dtype=bool)

    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distance = np.full((n, n), np.inf, dtype=float)
    distance[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distance, np.inf)

    values = distance[valid]
    positive = values[values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    spread = float(np.percentile(values, 75) -
                   np.percentile(values, 25))
    if not np.isfinite(spread) or spread <= 0.0:
        spread = scale

    temperatures = np.array(
        [
            max(0.45 * scale, 0.35 * spread),
            max(0.90 * scale, 0.70 * spread),
            max(1.80 * scale, 1.20 * spread),
        ],
        dtype=float,
    )

    eps = np.finfo(float).eps
    all_scores = []

    for temperature in temperatures:
        temperature = max(float(temperature), eps)

        kernel = np.zeros((n, n), dtype=float)
        normalized = np.minimum(distance / temperature, 60.0)
        kernel[valid] = np.exp(-normalized[valid])

        positive_kernel = kernel[valid]
        if positive_kernel.size:
            floor = max(float(np.max(positive_kernel)) * 1e-12, eps)
            kernel[valid] = np.maximum(kernel[valid], floor)

        balanced = kernel.copy()

        for _ in range(80):
            row_sum = balanced.sum(axis=1)
            row_sum = np.maximum(row_sum, eps)
            balanced /= row_sum[:, None]

            col_sum = balanced.sum(axis=0)
            col_sum = np.maximum(col_sum, eps)
            balanced /= col_sum[None, :]

            balanced[~valid] = 0.0
            np.fill_diagonal(balanced, 0.0)

        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

        maximum = float(np.max(balanced))
        if np.isfinite(maximum) and maximum > 0.0:
            balanced /= maximum

        all_scores.append(balanced)

    matching_score = np.ones((n, n), dtype=float)
    for score in all_scores:
        matching_score *= np.maximum(score, eps)

    matching_score = matching_score ** (1.0 / len(all_scores))
    matching_score[~valid] = 0.0
    np.fill_diagonal(matching_score, 0.0)

    row_mass = matching_score.sum(axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, eps)
    transition = matching_score / row_mass

    two_step = transition @ transition
    two_step = 0.5 * (two_step + two_step.T)

    support_values = two_step[valid]
    if support_values.size:
        support_scale = float(np.percentile(support_values, 90))
    else:
        support_scale = 1.0

    support_scale = max(support_scale, eps)
    cycle_support = np.zeros((n, n), dtype=float)
    cycle_support[valid] = np.tanh(
        two_step[valid] / support_scale
    )

    support_max = float(np.max(cycle_support))
    if np.isfinite(support_max) and support_max > 0.0:
        cycle_support /= support_max

    result = (
        np.maximum(matching_score, 0.0)
        * (
            0.78
            + 0.22 * np.maximum(cycle_support, 0.0)
        )
    )

    result = 0.5 * (result + result.T)
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)
    result = np.maximum(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
