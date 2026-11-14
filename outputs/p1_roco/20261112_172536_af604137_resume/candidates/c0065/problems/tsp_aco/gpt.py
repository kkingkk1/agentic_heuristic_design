import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d)
    valid &= ~np.eye(n, dtype=bool)
    valid &= d >= 0

    if not np.any(valid):
        return score

    work = np.where(valid, d, np.inf)

    ordered = np.sort(work, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else nearest * 1.8

    finite_nearest = np.isfinite(nearest)
    fallback = np.where(finite_nearest, nearest, 1.0)
    second = np.where(np.isfinite(second), second, fallback * 1.8)

    regret = np.maximum(second - fallback, 0.0)
    regret_scale = np.maximum(second, np.finfo(float).eps)
    node_regret = np.clip(regret / regret_scale, 0.0, 1.0)

    row_scale = np.where(
        finite_nearest,
        0.6 * fallback + 0.4 * second,
        1.0,
    )
    col_scale = row_scale.copy()

    pair_scale = np.sqrt(
        np.maximum(row_scale[:, None], np.finfo(float).eps)
        * np.maximum(col_scale[None, :], np.finfo(float).eps)
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=pair_scale > 0,
    )

    attractiveness = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    endpoint_regret = 1.0 + 0.60 * (
        node_regret[:, None] + node_regret[None, :]
    )

    row_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                np.maximum(row_scale[:, None], np.finfo(float).eps),
            ),
            0.0,
            50.0,
        )
    )
    col_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                np.maximum(col_scale[None, :], np.finfo(float).eps),
            ),
            0.0,
            50.0,
        )
    )

    exclusivity = 0.45 + 0.55 * np.sqrt(row_quality * col_quality)

    score = attractiveness * endpoint_regret * exclusivity
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_totals = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0,
    )

    return score
