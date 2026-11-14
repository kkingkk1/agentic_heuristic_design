import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

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

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    largest = float(np.max(observed)) if observed.size else safe_scale
    if not np.isfinite(largest):
        largest = safe_scale

    penalty = largest + 16.0 * safe_scale
    distances = np.where(valid, cleaned, penalty)
    np.fill_diagonal(distances, penalty)
    distances = 0.5 * (distances + distances.T)

    positive_distances = distances[valid]
    median_distance = (
        float(np.median(positive_distances))
        if positive_distances.size
        else safe_scale
    )

    temperature = max(0.24 * median_distance, 0.08 * safe_scale, eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / temperature, 60.0)
    )

    maximum = float(np.max(affinity))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    affinity /= maximum

    row_mass = affinity.sum(axis=1)
    row_preference = affinity / np.maximum(row_mass[:, None], eps)

    mutual_preference = np.sqrt(
        np.maximum(row_preference * row_preference.T, 0.0)
    )
    mutual_preference[~valid] = 0.0

    ordered = np.sort(
        np.where(valid, distances, penalty),
        axis=1,
    )
    nearest = ordered[:, 0]
    second = ordered[:, min(1, n - 1)]

    gap_scale = np.maximum(
        second - nearest,
        0.14 * safe_scale,
    )
    gap_scale = np.maximum(gap_scale, eps)

    local_preference = np.exp(
        np.clip(
            -(distances - nearest[:, None]) / gap_scale[:, None],
            -30.0,
            0.0,
        )
    )
    local_preference[~valid] = 0.0
    local_preference /= np.maximum(
        local_preference.max(axis=1, keepdims=True),
        eps,
    )

    mutual_local_preference = np.sqrt(
        np.maximum(
            local_preference * local_preference.T,
            0.0,
        )
    )
    mutual_local_preference[~valid] = 0.0

    regret = np.exp(
        np.clip(
            (second[:, None] - distances) / gap_scale[:, None],
            -24.0,
            24.0,
        )
    )
    regret[~valid] = 0.0

    mutual_regret = np.sqrt(
        np.maximum(regret * regret.T, 0.0)
    )
    mutual_regret[~valid] = 0.0

    bypass = np.full(
        (n, n),
        penalty,
        dtype=float,
    )

    block_size = max(1, min(n, 2_000_000 // max(n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        candidate = (
            distances[:, start:stop, None]
            + distances[None, start:stop, :]
        )
        bypass = np.minimum(
            bypass,
            np.min(candidate, axis=1),
        )

    np.fill_diagonal(bypass, penalty)

    bypass_gain = np.maximum(bypass - distances, 0.0)
    bypass_scale = max(0.68 * safe_scale, eps)

    irreplaceability = 1.0 - np.exp(
        -bypass_gain / bypass_scale
    )
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    scarcity = 1.0 / np.sqrt(
        np.maximum(row_mass, eps)
    )
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.70, 1.60)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )
    endpoint_scarcity[~valid] = 0.0

    local_signal = (
        0.42 * mutual_preference
        + 0.34 * mutual_local_preference
        + 0.24 * mutual_regret
    )

    structural_signal = (
        0.72 * irreplaceability
        + 0.28 * endpoint_scarcity
    )

    score = (
        np.maximum(affinity, eps) ** 0.90
        * np.maximum(0.25 + local_signal, eps) ** 0.95
        * np.maximum(0.35 + structural_signal, eps) ** 0.85
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

    balanced = score.copy()
    for _ in range(24):
        row_sum = balanced.sum(
            axis=1,
            keepdims=True,
        )
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.72 * score + 0.28 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
