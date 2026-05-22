"""
test_statistical_utils.py — 统计工具单元测试
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.evaluation.statistical_utils import (
    bootstrap_ci, per_class_ci, mcnemar_test, mcnemar_matrix, evaluate_with_ci
)


class TestBootstrapCI:
    def test_perfect_prediction(self):
        y = [0, 1, 2, 3, 4]
        acc, lo, hi = bootstrap_ci(y, y)
        assert acc == 1.0
        assert lo == 1.0
        assert hi == 1.0

    def test_worst_prediction(self):
        y_true = [0, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 1]
        acc, lo, hi = bootstrap_ci(y_true, y_pred)
        assert acc == 0.0

    def test_ci_contains_true_value(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 200)
        y_pred = y_true.copy()
        y_pred[:30] = (y_true[:30] + 1) % 5  # 85% accuracy
        acc, lo, hi = bootstrap_ci(y_true, y_pred)
        assert lo <= acc <= hi
        assert lo <= 0.85 <= hi

    def test_custom_metric(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 3]  # 80% accuracy
        metric = lambda yt, yp: np.mean(np.array(yt) == np.array(yp))
        acc, lo, hi = bootstrap_ci(y_true, y_pred, metric_fn=metric)
        assert abs(acc - 0.8) < 0.01


class TestMcNemar:
    def test_identical_predictions(self):
        y = [0, 1, 2, 3, 4]
        result = mcnemar_test(y, y, y)
        assert result["b01"] == 0
        assert result["b10"] == 0
        assert result["p_value"] == 1.0
        assert not result["significant"]

    def test_significant_difference(self):
        y_true = [0] * 100
        y_pred_a = [0] * 100  # perfect
        y_pred_b = [1] * 100  # all wrong
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert result["b01"] == 100
        assert result["b10"] == 0
        assert result["significant"]

    def test_no_significance(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 50)
        y_pred_a = y_true.copy()
        y_pred_b = y_true.copy()
        # 只有2个不同
        y_pred_a[0] = (y_true[0] + 1) % 5
        y_pred_b[1] = (y_true[1] + 1) % 5
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert not result["significant"]


class TestMcNemarMatrix:
    def test_three_methods(self):
        y_true = [0, 1, 2, 3, 4] * 20
        preds = {
            "Method_A": y_true.copy(),
            "Method_B": [(y + 1) % 5 for y in y_true],
            "Method_C": y_true.copy(),
        }
        result = mcnemar_matrix(y_true, preds)
        assert len(result["pairs"]) == 3
        assert "Method_A" in result["names"]


class TestEvaluateWithCI:
    def test_output_format(self):
        y_true = [0, 1, 2, 3, 4] * 20
        y_pred = y_true.copy()
        y_pred[:10] = [(y + 1) % 5 for y in y_true[:10]]
        report = evaluate_with_ci(y_true, y_pred, ["LM", "NET", "GIST", "EP", "LIP"])
        assert "Accuracy" in report
        assert "95% CI" in report
        assert "Macro F1" in report
