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

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    def local_scales(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ordered = np.sort(matrix, axis=1)
        nearest = ordered[:, 0]
        second = ordered[:, 1] if n > 2 else nearest * 2.0

        finite_nearest = np.isfinite(nearest)
        nearest_fallback = np.where(finite_nearest, nearest, 1.0)
        second = np.where(
            np.isfinite(second),
            second,
            nearest_fallback * 2.0,
        )

        regret = np.maximum(second - nearest_fallback, 0.0)
        regret_scale = np.maximum(second, eps)
        node_regret = np.clip(regret / regret_scale, 0.0, 1.0)

        scale = np.where(
            finite_nearest,
            0.5 * (nearest_fallback + second),
            1.0,
        )
        return scale, node_regret

    # Outgoing statistics are computed row-wise.
    outgoing_scale, outgoing_regret = local_scales(work)

    # Incoming statistics are computed on the transposed matrix.
    incoming_scale, incoming_regret = local_scales(work.T)

    pair_scale = np.sqrt(
        np.maximum(outgoing_scale[:, None], eps)
        * np.maximum(incoming_scale[None, :], eps)
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=pair_scale > 0,
    )
    attractiveness = np.exp(-np.clip(relative_distance, 0.0, 60.0))

    endpoint_regret = 1.0 + 0.45 * (
        outgoing_regret[:, None] + incoming_regret[None, :]
    )

    outgoing_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                np.maximum(outgoing_scale[:, None], eps),
            ),
            0.0,
            60.0,
        )
    )
    incoming_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                np.maximum(incoming_scale[None, :], eps),
            ),
            0.0,
            60.0,
        )
    )

    exclusivity = 0.55 + 0.45 * np.sqrt(
        outgoing_quality * incoming_quality
    )

    # Weak long-term continuation bonus:
    # destination j should retain usable outgoing options, while source i
    # should retain usable incoming alternatives. This is intentionally bounded.
    outgoing_options = np.max(
        np.where(valid, outgoing_quality, 0.0),
        axis=1,
    )
    incoming_options = np.max(
        np.where(valid, incoming_quality, 0.0),
        axis=0,
    )

    continuation = np.sqrt(
        np.clip(incoming_options[None, :], 0.0, 1.0)
        * np.clip(outgoing_options[:, None], 0.0, 1.0)
    )

    score = (
        attractiveness
        * endpoint_regret
        * exclusivity
        * (1.0 + 0.08 * continuation)
    )

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
