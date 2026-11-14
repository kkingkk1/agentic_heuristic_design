import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    d = np.where(np.isfinite(d), d, fallback)

    positive_values = d[d > 0]
    floor = (
        float(np.percentile(positive_values, 5)) * 1e-3
        if positive_values.size
        else 1e-12
    )
    floor = max(floor, 1e-12)

    # 基础距离先验：短边优先，并抑制极端距离造成的数值不稳定。
    safe_d = np.maximum(d, floor)
    inverse_distance = 1.0 / safe_d
    inverse_distance /= max(float(np.max(inverse_distance)), 1e-12)

    # 每个节点的局部距离尺度。
    local_scale = np.median(safe_d, axis=1)
    local_scale = np.maximum(local_scale, floor)

    # 局部尺度相似的边更可能连接相近的局部结构。
    scale_ratio = np.minimum(
        local_scale[:, None] / local_scale[None, :],
        local_scale[None, :] / local_scale[:, None],
    )
    scale_similarity = np.sqrt(np.clip(scale_ratio, 0.0, 1.0))

    # 邻居排名先验：边在两端节点的邻居排序越靠前，分数越高。
    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)

    # 互近邻强度：两端都把对方视为近邻时额外加分。
    mutual_neighbor = np.sqrt(rank_score * rank_score.T)

    # 对长尾距离使用指数衰减，避免仅依赖单一倒数尺度。
    distance_scale = np.median(positive_values) if positive_values.size else 1.0
    distance_scale = max(float(distance_scale), floor)
    exponential_score = np.exp(-safe_d / distance_scale)
    exponential_score /= max(float(np.max(exponential_score)), 1e-12)

    score = (
        0.42 * inverse_distance
        + 0.25 * rank_score
        + 0.18 * mutual_neighbor
        + 0.10 * scale_similarity
        + 0.05 * exponential_score
    )

    # TSP 回路中的边无方向，合并两个方向的证据。
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
