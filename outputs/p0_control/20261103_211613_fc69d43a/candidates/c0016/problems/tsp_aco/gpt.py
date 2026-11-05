import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square NumPy array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    safe_d = np.where(valid, d, np.nan)

    row_scale = np.nanmedian(safe_d, axis=1)
    finite_row_scale = row_scale[np.isfinite(row_scale)]

    fallback = (
        float(np.median(finite_row_scale))
        if finite_row_scale.size
        else 1.0
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        fallback,
    )

    normalized = np.divide(
        d,
        np.sqrt(row_scale[:, None] * row_scale[None, :]),
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    affinity = np.divide(
        1.0,
        normalized,
        out=np.zeros_like(d, dtype=float),
        where=valid & (normalized > 0),
    )

    row_best = np.max(affinity, axis=1)
    col_best = np.max(affinity, axis=0)

    row_competitiveness = np.divide(
        affinity,
        row_best[:, None],
        out=np.zeros_like(d, dtype=float),
        where=row_best[:, None] > 0,
    )
    col_competitiveness = np.divide(
        affinity,
        col_best[None, :],
        out=np.zeros_like(d, dtype=float),
        where=col_best[None, :] > 0,
    )

    indispensability = (1.0 - row_competitiveness) * (
        1.0 - col_competitiveness
    )

    score = affinity * (1.0 + indispensability)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
