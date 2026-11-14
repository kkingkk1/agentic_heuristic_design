import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric stochastic-sampling priors for TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    if positive.size:
        replacement = float(np.max(positive))
        global_center = float(np.median(positive))
        mad = float(np.median(np.abs(positive - global_center)))
    else:
        replacement = 1.0
        global_center = 1.0
        mad = 0.0

    replacement = max(replacement, np.finfo(float).eps)
    scale_floor = np.finfo(float).eps * max(1.0, replacement)

    d = np.where(finite, distances, replacement)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.nan)

    row_mean = np.nanmean(work, axis=1)
    row_mean = np.nan_to_num(
        row_mean,
        nan=global_center,
        posinf=replacement,
        neginf=0.0,
    )

    row_deviation = np.nanmean(
        np.abs(work - row_mean[:, None]),
        axis=1,
    )
    row_deviation = np.nan_to_num(
        row_deviation,
        nan=mad,
        posinf=replacement,
        neginf=0.0,
    )

    temperature = np.maximum(
        0.5 * row_deviation + 0.5 * mad,
        scale_floor,
    )

    logits = -d / temperature[:, None]
    logits = np.clip(logits, -60.0, 0.0)

    np.fill_diagonal(logits, -np.inf)
    row_max = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits - row_max)
    np.fill_diagonal(weights, 0.0)

    normalizer = np.sum(weights, axis=1, keepdims=True)
    probabilities = np.divide(
        weights,
        normalizer,
        out=np.zeros_like(weights),
        where=normalizer > 0.0,
    )

    mutual_preference = np.sqrt(
        np.maximum(probabilities, 0.0)
        * np.maximum(probabilities.T, 0.0)
    )

    robust_scale = max(global_center + mad, scale_floor)
    global_contrast = 1.0 / (
        1.0 + np.maximum(d - global_center, 0.0) / robust_scale
    )

    score = mutual_preference * (0.85 + 0.15 * global_contrast)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
