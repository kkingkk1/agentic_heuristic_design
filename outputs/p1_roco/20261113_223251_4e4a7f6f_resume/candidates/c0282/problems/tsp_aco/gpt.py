import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[forward_only] = raw[forward_only]
    cleaned[backward_only] = raw.T[backward_only]

    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    if not np.isfinite(maximum):
        maximum = scale

    invalid_cost = maximum + 24.0 * scale
    distances = np.where(valid, cleaned, invalid_cost)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, invalid_cost)

    valid_distances = distances[valid]
    temperature = float(np.median(valid_distances)) if valid_distances.size else scale
    temperature = max(0.30 * temperature, np.finfo(float).eps)

    affinity = np.exp(
        -np.clip(distances / temperature, 0.0, 80.0)
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if not np.isfinite(affinity_max) or affinity_max <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    affinity /= affinity_max

    ordered = np.sort(
        np.where(valid, distances, invalid_cost),
        axis=1,
    )
    nearest = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else nearest

    gap = np.maximum(second - nearest, 0.12 * scale)
    gap = np.maximum(gap, np.finfo(float).eps)

    rank_signal = np.exp(
        -np.clip(
            (distances - nearest[:, None]) / gap[:, None],
            0.0,
            24.0,
        )
    )
    rank_signal[~valid] = 0.0
    np.fill_diagonal(rank_signal, 0.0)

    rank_signal /= np.maximum(
        rank_signal.max(axis=1, keepdims=True),
        np.finfo(float).eps,
    )

    endpoint_preference = np.sqrt(
        np.maximum(rank_signal, 0.0)
        * np.maximum(rank_signal.T, 0.0)
    )

    local_choice_count = np.sum(
        rank_signal >= 0.35,
        axis=1,
        keepdims=True,
    ).astype(float)

    scarcity = 1.0 / np.sqrt(np.maximum(local_choice_count, 1.0))
    scarcity_pair = np.sqrt(scarcity * scarcity.T)

    replacement = np.full((n, n), invalid_cost, dtype=float)
    for k in range(n):
        candidate = distances[:, k, None] + distances[None, k, :]
        replacement = np.minimum(replacement, candidate)

    np.fill_diagonal(replacement, invalid_cost)

    bypass_gain = np.maximum(replacement - distances, 0.0)
    bypass_scale = max(0.75 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(
        -np.clip(bypass_gain / bypass_scale, 0.0, 32.0)
    )
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    distance_margin = np.maximum(second[:, None] - distances, 0.0)
    margin_signal = 1.0 - np.exp(
        -np.clip(distance_margin / gap[:, None], 0.0, 32.0)
    )
    margin_signal[~valid] = 0.0
    np.fill_diagonal(margin_signal, 0.0)

    margin_pair = np.sqrt(
        np.maximum(margin_signal, 0.0)
        * np.maximum(margin_signal.T, 0.0)
    )

    score = (
        0.38 * affinity
        + 0.24 * endpoint_preference
        + 0.16 * margin_pair
        + 0.14 * irreplaceability
        + 0.08 * scarcity_pair
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(24):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, np.finfo(float).eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.68 * score + 0.32 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    result_max = float(np.max(result))
    if np.isfinite(result_max) and result_max > 0.0:
        result /= result_max

    return result
