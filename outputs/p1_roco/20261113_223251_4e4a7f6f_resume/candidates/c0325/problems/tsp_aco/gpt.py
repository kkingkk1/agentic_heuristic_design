import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic stochastic-walk-based edge priors for TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0

    valid = (
        finite
        & nonnegative
        & finite.T
        & (raw.T >= 0.0)
        & ~diagonal
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distances, np.inf)

    values = distances[valid]
    positive = values[values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
    else:
        median = 1.0
        mad = 0.0

    if not np.isfinite(median) or median <= 0.0:
        median = 1.0

    scale = max(
        mad,
        0.08 * median,
        np.finfo(float).eps,
    )

    clean = np.where(valid, distances, median + 24.0 * scale)
    np.fill_diagonal(clean, median + 24.0 * scale)

    valid_counts = valid.sum(axis=1).astype(float)
    safe_counts = np.maximum(valid_counts, 1.0)

    ordered = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        ordered,
    ] = np.arange(n, dtype=float)

    local_rank = 1.0 - ranks / np.maximum(
        valid_counts[:, None] - 1.0,
        1.0,
    )
    local_rank = np.clip(local_rank, 0.0, 1.0)
    local_rank[~valid] = 0.0

    temperature = max(
        0.55 * median + 0.75 * scale,
        np.finfo(float).eps,
    )

    kernel = np.zeros((n, n), dtype=float)
    kernel[valid] = np.exp(
        -np.clip(
            (clean[valid] - np.min(clean[valid]))
            / temperature,
            0.0,
            50.0,
        )
    )

    row_mass = kernel.sum(axis=1)
    transition = kernel / np.maximum(
        row_mass[:, None],
        np.finfo(float).eps,
    )
    transition[~valid] = 0.0

    reverse_support = transition.T
    bidirectional = np.sqrt(
        np.maximum(
            transition * reverse_support,
            0.0,
        )
    )
    bidirectional[~valid] = 0.0

    two_step = transition @ transition
    two_step = np.maximum(two_step, 0.0)
    two_step[~valid] = 0.0

    two_step_max = np.max(two_step, axis=1, keepdims=True)
    closure = two_step / np.maximum(
        two_step_max,
        np.finfo(float).eps,
    )
    closure[~valid] = 0.0

    probabilities = transition[valid]
    if probabilities.size:
        entropy = -np.sum(
            transition * np.log(
                np.maximum(transition, np.finfo(float).eps)
            ),
            axis=1,
        )
        entropy /= np.log(
            np.maximum(valid_counts, 2.0)
        )
    else:
        entropy = np.ones(n, dtype=float)

    entropy = np.nan_to_num(
        entropy,
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    entropy = np.clip(entropy, 0.0, 1.0)

    certainty = 1.0 - entropy
    certainty_pair = np.sqrt(
        np.maximum(
            certainty[:, None] * certainty[None, :],
            0.0,
        )
    )
    certainty_pair[~valid] = 0.0

    edge_percentile = np.zeros((n, n), dtype=float)
    sorted_values = np.sort(values)
    positions = np.searchsorted(
        sorted_values,
        distances[valid],
        side="right",
    ).astype(float)
    edge_percentile[valid] = 1.0 - positions / max(
        float(sorted_values.size),
        1.0,
    )

    row_scale = np.maximum(
        np.mean(kernel, axis=1),
        np.finfo(float).eps,
    )
    node_activity = row_mass / np.maximum(
        np.max(row_mass),
        np.finfo(float).eps,
    )
    activity_pair = np.sqrt(
        np.maximum(
            node_activity[:, None] * node_activity[None, :],
            0.0,
        )
    )
    activity_pair[~valid] = 0.0

    local_component = (
        0.42 * np.sqrt(
            np.maximum(
                local_rank * local_rank.T,
                0.0,
            )
        )
        + 0.38 * bidirectional
        + 0.20 * certainty_pair
    )
    local_component[~valid] = 0.0

    global_component = (
        0.55 * edge_percentile
        + 0.25 * activity_pair
        + 0.20 * closure
    )
    global_component[~valid] = 0.0

    score = (
        np.maximum(local_component, 0.0) ** 1.35
        * np.maximum(global_component, 0.0) ** 0.85
        * np.maximum(0.25 + 0.75 * bidirectional, 0.0)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    row_total = score.sum(axis=1, keepdims=True)
    row_profile = score / np.maximum(
        row_total,
        np.finfo(float).eps,
    )

    profile_agreement = np.sqrt(
        np.maximum(
            row_profile * row_profile.T,
            0.0,
        )
    )
    profile_agreement[~valid] = 0.0

    result = (
        0.72 * score
        + 0.28 * profile_agreement
    )
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

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
