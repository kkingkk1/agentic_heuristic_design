import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge scores for a fixed ant-colony TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros(distances.shape, dtype=float)

    if n <= 1:
        return scores

    finite_positive = np.isfinite(distances) & (distances > 0)
    usable = distances[finite_positive]

    if usable.size == 0:
        return scores

    fallback = float(np.median(usable))

    transposed_valid = finite_positive.T
    both_valid = finite_positive & transposed_valid
    only_forward = finite_positive & ~transposed_valid
    only_reverse = ~finite_positive & transposed_valid

    symmetric_valid = finite_positive | transposed_valid
    symmetric_distances = np.full(distances.shape, fallback, dtype=float)

    symmetric_distances[both_valid] = (
        distances[both_valid] + distances.T[both_valid]
    ) * 0.5
    symmetric_distances[only_forward] = distances[only_forward]
    symmetric_distances[only_reverse] = distances.T[only_reverse]

    candidates = symmetric_distances.copy()
    candidates[~symmetric_valid] = np.nan
    np.fill_diagonal(candidates, np.nan)

    with np.errstate(all="ignore"):
        baselines = np.nanmedian(candidates, axis=1)

    baselines = np.where(
        np.isfinite(baselines) & (baselines > 0),
        baselines,
        fallback,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inverse_distance = 1.0 / symmetric_distances
        competitiveness = (
            0.5 * baselines[:, None] / symmetric_distances
            + 0.5 * baselines[None, :] / symmetric_distances
        )
        raw_scores = inverse_distance * (1.0 + competitiveness)

    valid_edges = symmetric_valid.copy()
    np.fill_diagonal(valid_edges, False)
    raw_scores = np.where(valid_edges & np.isfinite(raw_scores), raw_scores, 0.0)

    valid_values = raw_scores[valid_edges]
    if valid_values.size == 0:
        return scores

    minimum = float(valid_values.min())
    maximum = float(valid_values.max())

    if maximum > minimum:
        scores = (raw_scores - minimum) / (maximum - minimum)
    else:
        scores[valid_edges] = 1.0

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    return scores.astype(float, copy=False)
