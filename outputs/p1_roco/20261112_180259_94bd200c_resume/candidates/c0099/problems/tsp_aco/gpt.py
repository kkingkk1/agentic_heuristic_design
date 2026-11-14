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

    eps = np.finfo(float).eps
    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    work = np.where(valid, d, np.inf)

    def local_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)
        nearest = ordered[:, 0]

        if matrix.shape[1] > 2:
            second = ordered[:, 1]
        else:
            second = nearest * 2.0

        has_nearest = np.isfinite(nearest)
        nearest_safe = np.where(has_nearest, nearest, 1.0)
        second_safe = np.where(
            np.isfinite(second),
            second,
            nearest_safe * 2.0,
        )

        scale = np.where(
            has_nearest,
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

    outgoing_scale, outgoing_regret = local_statistics(work)
    incoming_scale, incoming_regret = local_statistics(work.T)

    outgoing_scale = np.maximum(outgoing_scale, eps)
    incoming_scale = np.maximum(incoming_scale, eps)

    outgoing_relative = np.divide(
        d,
        outgoing_scale[:, None],
        out=np.full_like(d, np.inf),
        where=outgoing_scale[:, None] > 0,
    )

    incoming_relative = np.divide(
        d,
        incoming_scale[None, :],
        out=np.full_like(d, np.inf),
        where=incoming_scale[None, :] > 0,
    )

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )

    pair_relative = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=pair_scale > 0,
    )

    proximity = np.exp(-np.clip(pair_relative, 0.0, 60.0))

    outgoing_quality = np.exp(
        -np.clip(outgoing_relative, 0.0, 60.0)
    )
    incoming_quality = np.exp(
        -np.clip(incoming_relative, 0.0, 60.0)
    )

    endpoint_regret = 1.0 + 0.45 * (
        outgoing_regret[:, None] + incoming_regret[None, :]
    )

    two_sided_quality = np.sqrt(
        outgoing_quality * incoming_quality
    )
    exclusivity = 0.55 + 0.45 * two_sided_quality

    outgoing_options = np.max(
        np.where(valid, outgoing_quality, 0.0),
        axis=1,
    )
    incoming_options = np.max(
        np.where(valid, incoming_quality, 0.0),
        axis=0,
    )

    continuation = np.sqrt(
        np.clip(outgoing_options[:, None], 0.0, 1.0)
        * np.clip(incoming_options[None, :], 0.0, 1.0)
    )

    score = (
        proximity
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
