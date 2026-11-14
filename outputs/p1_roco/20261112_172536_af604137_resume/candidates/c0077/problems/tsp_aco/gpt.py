import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = distances[valid]
    global_scale = float(np.median(positive))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    safe_distances = np.where(valid, distances, np.nan)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out_scale = np.nanmedian(safe_distances, axis=1)
        in_scale = np.nanmedian(safe_distances, axis=0)

        out_scale = np.where(
            np.isfinite(out_scale) & (out_scale > 0),
            out_scale,
            global_scale,
        )
        in_scale = np.where(
            np.isfinite(in_scale) & (in_scale > 0),
            in_scale,
            global_scale,
        )

        relative_out = distances / np.maximum(out_scale[:, None], eps)
        relative_in = distances / np.maximum(in_scale[None, :], eps)

        relative_out = np.where(valid, np.maximum(relative_out, eps), np.inf)
        relative_in = np.where(valid, np.maximum(relative_in, eps), np.inf)

        base = 1.0 / np.sqrt(relative_out * relative_in)

        safe_for_min = np.where(valid, distances, np.inf)

        out_best = np.min(safe_for_min, axis=1)
        in_best = np.min(safe_for_min, axis=0)

        if n >= 2:
            out_second = np.partition(safe_for_min, 1, axis=1)[:, 1]
            in_second = np.partition(safe_for_min, 1, axis=0)[1, :]
        else:
            out_second = np.full(n, np.inf)
            in_second = np.full(n, np.inf)

        out_count = np.sum(valid, axis=1)
        in_count = np.sum(valid, axis=0)

        out_gap = np.where(
            np.isfinite(out_second) & (out_count >= 2),
            np.clip(
                (out_second - out_best)
                / np.maximum(out_second, eps),
                0.0,
                1.0,
            ),
            0.0,
        )
        in_gap = np.where(
            np.isfinite(in_second) & (in_count >= 2),
            np.clip(
                (in_second - in_best)
                / np.maximum(in_second, eps),
                0.0,
                1.0,
            ),
            0.0,
        )

        out_regret = np.where(
            valid,
            np.clip(
                (out_second[:, None] - distances)
                / np.maximum(out_second[:, None], eps),
                0.0,
                1.0,
            ),
            0.0,
        )
        in_regret = np.where(
            valid,
            np.clip(
                (in_second[None, :] - distances)
                / np.maximum(in_second[None, :], eps),
                0.0,
                1.0,
            ),
            0.0,
        )

        gap_factor = (
            1.0
            + 0.04
            * np.sqrt(
                out_gap[:, None]
                * in_gap[None, :]
            )
            * np.sqrt(out_regret * in_regret)
        )

        score = np.where(valid, base * gap_factor, 0.0)

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    score = score / row_max

    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
