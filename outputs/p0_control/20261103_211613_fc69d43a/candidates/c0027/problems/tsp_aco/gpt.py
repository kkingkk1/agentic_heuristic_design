import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return an edge-score matrix for a fixed ant-colony TSP solver.

    Larger scores indicate edges that are more attractive for minimizing tour
    length. The score combines scale-normalized inverse distance with a
    two-sided local competition advantage.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(distances, dtype=float)

    safe_distances = np.where(valid, distances, np.nan)

    row_scale = np.nanmedian(
        np.where(valid, safe_distances, np.nan),
        axis=1
    )
    col_scale = np.nanmedian(
        np.where(valid, safe_distances, np.nan),
        axis=0
    )

    global_scale = np.nanmedian(safe_distances)
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0.0),
        row_scale,
        global_scale
    )
    col_scale = np.where(
        np.isfinite(col_scale) & (col_scale > 0.0),
        col_scale,
        global_scale
    )

    row_ratio = np.divide(
        distances,
        row_scale[:, None],
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid
    )
    col_ratio = np.divide(
        distances,
        col_scale[None, :],
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid
    )

    row_base = 1.0 / (1.0 + row_ratio)
    col_base = 1.0 / (1.0 + col_ratio)

    row_best = np.min(
        np.where(valid, distances, np.inf),
        axis=1
    )
    col_best = np.min(
        np.where(valid, distances, np.inf),
        axis=0
    )

    row_advantage = np.divide(
        row_best[:, None],
        distances,
        out=np.zeros_like(distances, dtype=float),
        where=valid & (distances > 0.0)
    )
    col_advantage = np.divide(
        col_best[None, :],
        distances,
        out=np.zeros_like(distances, dtype=float),
        where=valid & (distances > 0.0)
    )

    zero_edges = valid & (distances == 0.0)
    row_advantage[zero_edges] = 1.0
    col_advantage[zero_edges] = 1.0

    base_score = np.sqrt(row_base * col_base)
    competition_score = np.sqrt(row_advantage * col_advantage)

    scores = base_score * (0.5 + 0.5 * competition_score)
    scores = np.where(valid, scores, 0.0)
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores = scores / maximum

    return scores
