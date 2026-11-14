import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directional edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d)
    valid &= d >= 0
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    work = np.where(valid, d, np.inf)
    eps = np.finfo(float).eps

    outgoing_sorted = np.sort(work, axis=1)
    out_nearest = outgoing_sorted[:, 0]
    out_second = outgoing_sorted[:, 1] if n > 2 else out_nearest * 2.0

    incoming_sorted = np.sort(work, axis=0)
    in_nearest = incoming_sorted[0, :]
    in_second = incoming_sorted[1, :] if n > 2 else in_nearest * 2.0

    def complete_scale(nearest: np.ndarray, second: np.ndarray):
        finite_nearest = np.isfinite(nearest)
        nearest_safe = np.where(finite_nearest, nearest, 1.0)
        second_safe = np.where(
            np.isfinite(second),
            second,
            nearest_safe * 2.0,
        )
        scale = np.where(
            finite_nearest,
            0.5 * (nearest_safe + second_safe),
            1.0,
        )
        regret = np.clip(
            (second_safe - nearest_safe)
            / np.maximum(second_safe, eps),
            0.0,
            1.0,
        )
        return scale, regret

    out_scale, out_regret = complete_scale(out_nearest, out_second)
    in_scale, in_regret = complete_scale(in_nearest, in_second)

    pair_scale = np.sqrt(
        np.maximum(out_scale[:, None], eps)
        * np.maximum(in_scale[None, :], eps)
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=pair_scale > 0,
    )
    proximity = np.exp(-np.clip(relative_distance, 0.0, 60.0))

    out_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                np.maximum(out_scale[:, None], eps),
            ),
            0.0,
            60.0,
        )
    )
    in_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                np.maximum(in_scale[None, :], eps),
            ),
            0.0,
            60.0,
        )
    )

    endpoint_regret = 1.0 + 0.55 * (
        out_regret[:, None] + in_regret[None, :]
    )

    exclusivity = 0.50 + 0.50 * np.sqrt(out_quality * in_quality)

    out_rank = np.divide(
        d,
        np.maximum(out_scale[:, None], eps),
    )
    in_rank = np.divide(
        d,
        np.maximum(in_scale[None, :], eps),
    )
    rank_bonus = 1.0 + 0.08 * np.exp(
        -np.clip(0.5 * (out_rank + in_rank), 0.0, 60.0)
    )

    score = proximity * endpoint_regret * exclusivity * rank_bonus
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
