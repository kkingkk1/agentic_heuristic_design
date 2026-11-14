import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative O(n^2) edge-selection priors for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = distances[valid & (distances > 0.0)]

    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        scale = max(median, 1.4826 * mad, eps)
    else:
        scale = 1.0

    safe = np.where(valid, np.maximum(distances, 0.0), np.inf)
    normalized = np.minimum(safe / scale, 40.0)

    attraction = (
        0.70 * np.exp(-normalized)
        + 0.30 / (1.0 + normalized)
    )
    attraction[~valid] = 0.0

    row_best = np.min(safe, axis=1, keepdims=True)
    col_best = np.min(safe, axis=0, keepdims=True)

    row_gap = np.minimum(np.maximum(safe - row_best, 0.0) / scale, 8.0)
    col_gap = np.minimum(np.maximum(safe - col_best, 0.0) / scale, 8.0)

    competition = 1.0 / (1.0 + 0.40 * row_gap + 0.40 * col_gap)
    competition[~valid] = 0.0

    scores = attraction * competition

    if n >= 3:
        row_two = np.partition(safe, 1, axis=1)[:, 1][:, None]
        col_two = np.partition(safe, 1, axis=0)[1, :][None, :]

        finite_alternatives = np.isfinite(row_two) & np.isfinite(col_two)

        row_advantage = np.maximum(row_two - safe, 0.0) / scale
        col_advantage = np.maximum(col_two - safe, 0.0) / scale

        scarcity_bonus = 1.0 + 0.16 * np.tanh(
            np.minimum(row_advantage, 6.0)
            + np.minimum(col_advantage, 6.0)
        )
        scarcity_bonus[~finite_alternatives] = 1.0
        scarcity_bonus[~valid] = 1.0
        scores *= scarcity_bonus

        # Restricted two-hop direct-edge gain using at most two local candidates.
        row_candidates = np.argpartition(safe, 1, axis=1)[:, :2]
        col_candidates = np.argpartition(safe.T, 1, axis=1)[:, :2]

        row_first = row_candidates[:, 0][:, None]
        row_second = row_candidates[:, 1][:, None]
        row_choice = np.where(row_first != np.arange(n)[:, None], row_first, row_second)
        row_choice = np.where(row_choice != np.arange(n)[None, :], row_choice, row_second)

        col_first = col_candidates[:, 0][None, :]
        col_second = col_candidates[:, 1][None, :]
        col_choice = np.where(col_first != np.arange(n)[None, :], col_first, col_second)
        col_choice = np.where(col_choice != np.arange(n)[:, None], col_choice, col_second)

        rows = np.arange(n)[:, None]
        cols = np.arange(n)[None, :]

        row_leg = safe[rows, row_choice]
        row_return = safe[row_choice, cols]
        col_leg = safe[rows, col_choice]
        col_return = safe[col_choice, cols]

        row_two_hop = row_leg + row_return
        col_two_hop = col_leg + col_return
        two_hop = np.minimum(row_two_hop, col_two_hop)

        two_hop_gain = np.zeros_like(safe)
        usable_hop = valid & np.isfinite(two_hop)
        two_hop_gain[usable_hop] = np.tanh(
            np.clip(
                (two_hop[usable_hop] - safe[usable_hop]) / scale,
                -6.0,
                6.0,
            )
        )

        scores *= 1.0 + 0.08 * two_hop_gain

    scores[~valid] = 0.0

    balanced = scores.copy()
    for _ in range(2):
        row_sum = np.sum(balanced, axis=1, keepdims=True)
        row_factor = 1.0 / np.sqrt(np.maximum(row_sum, eps))
        row_reference = float(np.median(row_factor[np.isfinite(row_factor)]))
        row_factor = np.clip(
            row_factor / max(row_reference, eps),
            0.86,
            1.16,
        )
        balanced *= row_factor

        col_sum = np.sum(balanced, axis=0, keepdims=True)
        col_factor = 1.0 / np.sqrt(np.maximum(col_sum, eps))
        col_reference = float(np.median(col_factor[np.isfinite(col_factor)]))
        col_factor = np.clip(
            col_factor / max(col_reference, eps),
            0.86,
            1.16,
        )
        balanced *= col_factor
        balanced[~valid] = 0.0

    scores = 0.84 * scores + 0.16 * balanced
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
