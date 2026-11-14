import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic-flow-based edge indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)
    symmetric_valid = valid & valid.T

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    positive_work = work[np.isfinite(work) & (work > 0.0)]
    global_scale = (
        float(np.median(positive_work))
        if positive_work.size
        else fallback
    )
    global_scale = max(global_scale, eps)

    finite_rows = np.where(np.isfinite(work), work, global_scale)
    row_mean = np.mean(finite_rows, axis=1)
    row_std = np.std(finite_rows, axis=1)

    temperature = global_scale * (
        0.65 + 0.35 * row_std / (row_mean + global_scale)
    )
    temperature = np.maximum(temperature, eps)

    logits = -work / temperature[:, None]
    row_max = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.clip(logits - row_max, -60.0, 0.0))
    weights[~np.isfinite(weights)] = 0.0
    np.fill_diagonal(weights, 0.0)

    row_sum = np.sum(weights, axis=1, keepdims=True)
    transition = weights / np.maximum(row_sum, eps)
    transition = np.nan_to_num(
        transition,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mutual_flow = np.sqrt(
        np.maximum(transition * transition.T, 0.0)
    )

    two_step_flow = transition @ transition
    two_step_flow = np.maximum(two_step_flow, 0.0)
    alternative_flow = np.sqrt(
        two_step_flow * two_step_flow.T
    )

    flow_ratio = alternative_flow / (mutual_flow + eps)
    exclusivity = 1.0 / (1.0 + np.clip(flow_ratio, 0.0, 50.0))

    detour_gain = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if n > 2:
                detours = d[i, :] + d[:, j]
                allowed = np.ones(n, dtype=bool)
                allowed[i] = False
                allowed[j] = False
                best_detour = float(np.min(detours[allowed]))
            else:
                best_detour = np.inf

            direct = float(d[i, j])
            if np.isfinite(best_detour):
                gain = (best_detour - direct) / (
                    best_detour + direct + eps
                )
                gain = float(np.clip(gain, 0.0, 1.0))
            else:
                gain = 0.0

            detour_gain[i, j] = gain
            detour_gain[j, i] = gain

    score = (
        np.power(np.maximum(mutual_flow, 0.0), 0.72)
        * (0.72 + 0.28 * exclusivity)
        * (0.86 + 0.14 * detour_gain)
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
