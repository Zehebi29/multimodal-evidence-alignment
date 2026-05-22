"""
statistical_utils.py — 统计严谨性工具模块
==========================================
提供 bootstrap 置信区间和 McNemar 检验，用于论文级实验评估。

用法:
    from statistical_utils import bootstrap_ci, mcnemar_test, evaluate_with_ci
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter


def bootstrap_ci(
    y_true: List,
    y_pred: List,
    metric_fn=None,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap 置信区间估计。

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        metric_fn: 评估函数，默认为 accuracy
        n_bootstrap: bootstrap 采样次数
        ci: 置信水平 (默认 95%)
        seed: 随机种子

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if metric_fn is None:
        metric_fn = lambda yt, yp: np.mean(yt == yp)

    point = metric_fn(y_true, y_pred)
    boot_stats = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        boot_stats.append(metric_fn(y_true[idx], y_pred[idx]))

    alpha = (1 - ci) / 2 * 100
    lower = np.percentile(boot_stats, alpha)
    upper = np.percentile(boot_stats, 100 - alpha)

    return point, lower, upper


def per_class_ci(
    y_true: List,
    y_pred: List,
    class_names: List[str],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    每个类别的 Precision/Recall/F1 的 bootstrap CI。

    Returns:
        {class_name: {"precision": (val, lo, hi), "recall": ..., "f1": ...}}
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rng = np.random.RandomState(seed)
    results = {}

    for cls_idx, cls_name in enumerate(class_names):
        metrics = {"precision": [], "recall": [], "f1": []}

        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_true), len(y_true), replace=True)
            yt, yp = y_true[idx], y_pred[idx]

            tp = np.sum((yt == cls_idx) & (yp == cls_idx))
            fp = np.sum((yt != cls_idx) & (yp == cls_idx))
            fn = np.sum((yt == cls_idx) & (yp != cls_idx))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            metrics["precision"].append(prec)
            metrics["recall"].append(rec)
            metrics["f1"].append(f1)

        alpha = (1 - ci) / 2 * 100
        results[cls_name] = {}
        for m_name, m_vals in metrics.items():
            pt = np.mean(m_vals)
            lo = np.percentile(m_vals, alpha)
            hi = np.percentile(m_vals, 100 - alpha)
            results[cls_name][m_name] = (pt, lo, hi)

    return results


def mcnemar_test(y_true: List, y_pred_a: List, y_pred_b: List) -> Dict:
    """
    McNemar 检验：比较两个分类器是否有显著差异。

    H0: 两个分类器的错误率相同。
    若 p < 0.05，拒绝 H0，认为两者有显著差异。

    Args:
        y_true: 真实标签
        y_pred_a: 方法 A 的预测
        y_pred_b: 方法 B 的预测

    Returns:
        {"b01": int, "b10": int, "statistic": float, "p_value": float, "significant": bool}
    """
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    # b01: A对B错, b10: A错B对
    b01 = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    b10 = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))

    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "statistic": 0.0, "p_value": 1.0, "significant": False}

    # 使用连续性校正的 McNemar 检验
    statistic = (abs(b01 - b10) - 1) ** 2 / n if n > 0 else 0.0

    # 近似卡方检验 (df=1)
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "b01": int(b01),
        "b10": int(b10),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05
    }


def mcnemar_matrix(
    y_true: List,
    predictions: Dict[str, List],
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    多方法两两 McNemar 检验矩阵。

    Args:
        y_true: 真实标签
        predictions: {"method_name": y_pred, ...}

    Returns:
        {"pairs": {(name_a, name_b): mcnemar_result, ...}, "matrix": p_value_matrix}
    """
    names = list(predictions.keys())
    n = len(names)
    pairs = {}
    matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            result = mcnemar_test(y_true, predictions[names[i]], predictions[names[j]])
            pairs[(names[i], names[j])] = result
            matrix[i][j] = result["p_value"]
            matrix[j][i] = result["p_value"]

    return {"pairs": pairs, "matrix": matrix, "names": names}


def evaluate_with_ci(
    y_true: List,
    y_pred: List,
    class_names: List[str],
    method_name: str = "Method",
    n_bootstrap: int = 2000
) -> str:
    """
    完整评估：准确率 + CI + 每类别 CI，返回格式化字符串。

    Args:
        y_true: 真实标签
        y_pred: 预测标签（整数索引）
        class_names: 类别名称列表
        method_name: 方法名称

    Returns:
        格式化的评估报告字符串
    """
    lines = [f"\n{'='*60}", f"  {method_name}", f"{'='*60}"]

    # 整体准确率 + CI
    acc, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, n_bootstrap=n_bootstrap)
    lines.append(f"  Accuracy: {acc*100:.1f}% (95% CI: [{acc_lo*100:.1f}%, {acc_hi*100:.1f}%])")

    # 每类别
    per_cls = per_class_ci(y_true, y_pred, class_names, n_bootstrap=n_bootstrap)
    lines.append(f"\n  {'Class':<10} {'Prec':>14} {'Recall':>14} {'F1':>14}")
    lines.append(f"  {'-'*52}")

    macro_f1s = []
    for cls in class_names:
        p = per_cls[cls]["precision"]
        r = per_cls[cls]["recall"]
        f = per_cls[cls]["f1"]
        macro_f1s.append(f[0])
        lines.append(f"  {cls:<10} {p[0]*100:5.1f}% [{p[1]*100:.1f},{p[2]*100:.1f}]"
                     f" {r[0]*100:5.1f}% [{r[1]*100:.1f},{r[2]*100:.1f}]"
                     f" {f[0]*100:5.1f}% [{f[1]*100:.1f},{f[2]*100:.1f}]")

    macro_f1 = np.mean(macro_f1s)
    lines.append(f"\n  Macro F1: {macro_f1*100:.1f}%")

    return "\n".join(lines)
