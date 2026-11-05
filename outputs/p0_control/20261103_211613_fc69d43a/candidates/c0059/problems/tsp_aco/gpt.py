import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge scores for a fixed ant-colony TSP solver."""
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
    safe_distances = np.where(finite_positive, distances, fallback)

    off_diagonal = ~np.eye(n, dtype=bool)
    candidates = np.where(off_diagonal, safe_distances, np.nan)

    row_baseline = np.nanmedian(candidates, axis=1)
    row_baseline = np.where(
        np.isfinite(row_baseline) & (row_baseline > 0),
        row_baseline,
        fallback,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        reciprocal_distance = np.divide(
            1.0,
            safe_distances,
            out=np.zeros_like(safe_distances),
            where=safe_distances > 0,
        )
        endpoint_competitiveness = 0.5 * (
            np.divide(
                row_baseline[:, None],
                safe_distances,
                out=np.zeros_like(safe_distances),
                where=safe_distances > 0,
            )
            + np.divide(
                row_baseline[None, :],
                safe_distances,
                out=np.zeros_like(safe_distances),
                where=safe_distances > 0,
            )
        )

    raw_scores = reciprocal_distance * (1.0 + endpoint_competitiveness)
    valid_edges = finite_positive & off_diagonal
    raw_scores = np.where(valid_edges, raw_scores, 0.0)

    valid_values = raw_scores[valid_edges]
    if valid_values.size == 0:
        return scores

    minimum = float(valid_values.min())
    maximum = float(valid_values.max())

    if maximum > minimum:
        scores = np.where(
            valid_edges,
            (raw_scores - minimum) / (maximum - minimum),
            0.0,
        )
    else:
        scores = np.where(valid_edges, 1.0, 0.0)

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    return scores
