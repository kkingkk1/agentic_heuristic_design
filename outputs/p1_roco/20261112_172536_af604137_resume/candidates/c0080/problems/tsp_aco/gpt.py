import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return non-negative directed edge desirability indicators for TSP sampling."""
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

    positive = distances[valid]
    global_scale = float(np.median(positive))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    safe = np.where(valid, distances, np.nan)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out_scale = np.nanmedian(safe, axis=1)
        in_scale = np.nanmedian(safe, axis=0)

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

        eps = np.finfo(float).eps

        relative_out = distances / np.maximum(out_scale[:, None], eps)
        relative_in = distances / np.maximum(in_scale[None, :], eps)

        relative_out = np.where(valid, np.maximum(relative_out, eps), np.inf)
        relative_in = np.where(valid, np.maximum(relative_in, eps), np.inf)

        base = 1.0 / np.sqrt(relative_out * relative_in)

        safe_distances = np.where(valid, distances, np.inf)

        out_best = np.min(safe_distances, axis=1)
        in_best = np.min(safe_distances, axis=0)

        out_second = np.partition(safe_distances, 1, axis=1)[:, 1]
        in_second = np.partition(safe_distances, 1, axis=0)[1, :]

        out_count = np.sum(valid, axis=1)
        in_count = np.sum(valid, axis=0)

        out_gap = np.where(
            (out_count >= 2) & np.isfinite(out_second),
            np.clip((out_second - out_best) / np.maximum(out_second, eps), 0.0, 1.0),
            0.0,
        )
        in_gap = np.where(
            (in_count >= 2) & np.isfinite(in_second),
            np.clip((in_second - in_best) / np.maximum(in_second, eps), 0.0, 1.0),
            0.0,
        )

        out_advantage = np.where(
            valid & np.isfinite(out_second[:, None]),
            np.clip(
                (out_second[:, None] - distances)
                / np.maximum(out_second[:, None], eps),
                0.0,
                1.0,
            ),
            0.0,
        )
        in_advantage = np.where(
            valid & np.isfinite(in_second[None, :]),
            np.clip(
                (in_second[None, :] - distances)
                / np.maximum(in_second[None, :], eps),
                0.0,
                1.0,
            ),
            0.0,
        )

        confidence = np.sqrt(
            out_gap[:, None] * in_gap[None, :]
        ) * np.sqrt(out_advantage * in_advantage)

        score = np.where(
            valid,
            base * (1.0 + 0.03 * confidence),
            0.0,
        )

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    score /= row_max

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return score
