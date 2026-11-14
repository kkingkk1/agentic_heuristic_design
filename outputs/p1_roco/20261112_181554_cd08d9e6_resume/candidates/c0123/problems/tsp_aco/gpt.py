import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    def regret_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ordered = np.sort(matrix, axis=1)
        nearest = ordered[:, 0]

        if matrix.shape[1] > 2:
            second = ordered[:, 1]
        else:
            second = nearest * 2.0

        has_nearest = np.isfinite(nearest)
        nearest_fallback = np.where(has_nearest, nearest, 1.0)
        second = np.where(
            np.isfinite(second),
            second,
            nearest_fallback * 2.0,
        )

        scale = 0.5 * (nearest_fallback + second)
        regret = np.maximum(second - nearest_fallback, 0.0)
        regret = regret / np.maximum(second, eps)
        regret = np.clip(regret, 0.0, 1.0)

        return scale, regret

    outgoing_scale, _ = regret_scale(work)
    incoming_scale, incoming_regret = regret_scale(work.T)

    relative_outgoing = np.divide(
        d,
        np.maximum(outgoing_scale[:, None], eps),
        out=np.full_like(d, np.inf, dtype=float),
        where=outgoing_scale[:, None] > 0,
    )

    attractiveness = np.exp(-np.clip(relative_outgoing, 0.0, 60.0))

    destination_priority = 1.0 + 0.30 * incoming_regret[None, :]

    incoming_relative = np.divide(
        d,
        np.maximum(incoming_scale[None, :], eps),
        out=np.full_like(d, np.inf, dtype=float),
        where=incoming_scale[None, :] > 0,
    )

    incoming_quality = np.exp(
        -np.clip(0.15 * incoming_relative, 0.0, 10.0)
    )

    score = attractiveness * destination_priority * incoming_quality
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
