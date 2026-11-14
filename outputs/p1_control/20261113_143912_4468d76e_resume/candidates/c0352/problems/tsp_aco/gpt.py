import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = np.isfinite(d) & (d > 0)
    valid_edges = positive & positive.T

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[positive]

    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe_distance = np.where(
        np.isfinite(d),
        np.maximum(d, eps),
        fallback_distance * 10.0,
    )

    # 用分区选择代替完整排序，提取每个端点的局部尺度。
    kth = min(1, n - 2)
    partitioned = np.partition(safe_distance, kth=kth, axis=1)
    local_scale = partitioned[:, kth]

    row_counts = np.sum(positive, axis=1)
    row_medians = np.full(n, global_scale, dtype=float)

    for i in np.flatnonzero(row_counts > 0):
        row_values = d[i, positive[i]]
        if row_values.size:
            row_medians[i] = np.median(row_values)

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale < fallback_distance * 10.0),
        local_scale,
        np.where(row_counts > 0, row_medians, fallback_distance),
    )
    local_scale = np.maximum(local_scale, eps)

    relative_length = safe_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    # 互惠邻近排名。
    order = np.argsort(safe_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank /= max(n - 1, 1)

    mutual_rank = rank + rank.T

    # 端点稀缺性：局部尺度越大，说明该端点可用的短边越少。
    scarcity = np.sqrt(
        np.clip(local_scale / global_scale, 0.25, 4.0)
    )
    scarcity = 1.0 + 0.16 * np.tanh(scarcity - 1.0)
    edge_scarcity = np.sqrt(scarcity[:, None] * scarcity[None, :])

    # 局部尺度不匹配惩罚，避免将异常长边直接连接到密集局部区域。
    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None:]
    balance = 1.0 / (
        1.0 + 0.70 * scale_difference / (scale_sum + eps)
    )
    balance = 0.80 + 0.20 * balance

    # 采用对数域组合，减少多个指数因子连续相乘造成的下溢。
    log_score = (
        -1.12 * np.log1p(relative_length)
        -2.75 * mutual_rank
        -0.24 * safe_distance / global_scale
        +np.log(0.68 + 0.32 * np.exp(
            -0.24 * safe_distance / global_scale
        ))
        +np.log(edge_scarcity)
        +np.log(balance)
    )

    log_score[~valid_edges] = -np.inf
    log_score = 0.5 * (log_score + log_score.T)

    finite_scores = np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = np.max(log_score[finite_scores])
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(log_score[finite_scores] - maximum)

    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
