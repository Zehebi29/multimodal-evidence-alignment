"""
Majority Vote Baseline — 病例级聚合

最简单的 baseline：对每个病例，统计所有阳性帧的类别投票，
票数最多的类别作为病例级预测。

输入: /data/eus/results/smt_v1/{case_id}.json (帧级结果)
输出: 病例级预测 + 评估指标

Multimodal Evidence Alignment (MEA) — Stage 1 Baseline
"""

import json
import os
from collections import Counter, defaultdict
from typing import Optional


# ── 类别定义 ──
CLASS_NAMES = {0: "LM", 1: "NET", 2: "GIST", 3: "EP", 4: "LIP"}
CONF_THRESHOLD = 0.5  # 低于此置信度的检测不计入投票


def load_case_result(result_path: str) -> dict:
    """加载单个病例的帧级结果"""
    with open(result_path) as f:
        return json.load(f)


def aggregate_majority_vote(case_result: dict, conf_threshold: float = CONF_THRESHOLD) -> dict:
    """
    Majority Vote 聚合
    
    规则：
    1. 遍历所有帧，每帧取最高置信度的检测（> threshold）
    2. 每帧投一票给该类别
    3. 票数最多的类别为病例级预测
    4. 置信度 = 该类别所有帧的平均置信度
    
    返回:
        case_id, dominant_class, confidence, consistency,
        class_votes, total_frames, positive_frames, frame_details
    """
    case_id = case_result["case_id"]
    frames = case_result["frames"]
    
    votes = Counter()        # 类别票数
    confs = defaultdict(list)  # 每类别的置信度列表
    frame_details = []       # 每帧详情
    
    for frame in frames:
        frame_id = frame["frame_id"]
        detections = frame.get("detections", [])
        
        # 取最高置信度的检测
        best_det = None
        for det in detections:
            conf = det["conf"]
            if conf >= conf_threshold:
                if best_det is None or conf > best_det["conf"]:
                    best_det = det
        
        if best_det:
            raw_class = best_det["class"]
            # 兼容两种格式：数字 ID ("0") 或名字 ("GIST")
            if raw_class.isdigit():
                cls_name = CLASS_NAMES.get(int(raw_class), f"UNK_{raw_class}")
            else:
                cls_name = raw_class
            conf = best_det["conf"]
            
            votes[cls_name] += 1
            confs[cls_name].append(conf)
            frame_details.append({
                "frame_id": frame_id,
                "vote": cls_name,
                "conf": conf,
                "bbox": best_det["bbox"],
            })
        else:
            frame_details.append({
                "frame_id": frame_id,
                "vote": None,
                "conf": 0.0,
                "bbox": None,
            })
    
    # 病例级预测
    total_frames = len(frames)
    positive_frames = sum(1 for f in frame_details if f["vote"] is not None)
    
    if votes:
        dominant_class = votes.most_common(1)[0][0]
        dominant_count = votes[dominant_class]
        confidence = sum(confs[dominant_class]) / len(confs[dominant_class])
        consistency = dominant_count / positive_frames if positive_frames > 0 else 0.0
    else:
        dominant_class = None
        confidence = 0.0
        consistency = 0.0
    
    return {
        "case_id": case_id,
        "dataset_version": case_result.get("dataset_version", "unknown"),
        "dominant_class": dominant_class,
        "confidence": round(confidence, 4),
        "consistency": round(consistency, 4),
        "class_votes": dict(votes),
        "total_frames": total_frames,
        "positive_frames": positive_frames,
        "frame_details": frame_details,
    }


def run_majority_vote_batch(results_dir: str, output_path: Optional[str] = None, conf_threshold: float = CONF_THRESHOLD) -> list:
    """
    对目录下所有病例运行 majority vote 聚合
    
    返回: 所有病例的聚合结果列表
    """
    case_files = sorted([
        f for f in os.listdir(results_dir)
        if f.endswith(".json")
    ])
    
    all_results = []
    for fname in case_files:
        case_path = os.path.join(results_dir, fname)
        case_result = load_case_result(case_path)
        agg = aggregate_majority_vote(case_result, conf_threshold=conf_threshold)
        all_results.append(agg)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"✅ 保存聚合结果: {output_path} ({len(all_results)} 病例)")
    
    return all_results


def evaluate_against_ground_truth(
    aggregated: list,
    cases_dir: str,
) -> dict:
    """
    与真实诊断对比评估
    
    返回: accuracy, per-class metrics, confusion matrix
    """
    DIAG_MAP = {
        "平滑肌瘤": "LM", "Leiomyoma": "LM",
        "GIST": "GIST", "胃肠道间质瘤": "GIST",
        "神经内分泌肿瘤": "NET", "NET": "NET",
        "异位胰腺": "EP", "Ectopic Pancreas": "EP",
        "脂肪瘤": "LIP", "Lipoma": "LIP",
    }
    
    confusion = defaultdict(lambda: defaultdict(int))
    skipped = 0
    
    for agg in aggregated:
        case_id = agg["case_id"]
        meta_path = os.path.join(cases_dir, case_id, "metadata.json")
        if not os.path.exists(meta_path):
            skipped += 1
            continue
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        gt_diag = meta["pathology_ground_truth"]["diagnosis"]
        gt_cls = DIAG_MAP.get(gt_diag)
        if not gt_cls:
            skipped += 1
            continue
        
        ai_cls = agg["dominant_class"] or "NONE"
        confusion[gt_cls][ai_cls] += 1
    
    # 计算指标
    classes = ["LM", "GIST", "NET", "EP", "LIP"]
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[c].get(c, 0) for c in classes)
    
    per_class = {}
    for cls in classes:
        tp = confusion[cls].get(cls, 0)
        fp = sum(confusion[g].get(cls, 0) for g in classes) - tp
        fn = sum(confusion[cls].values()) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[cls].values()),
        }
    
    return {
        "method": "majority_vote",
        "conf_threshold": CONF_THRESHOLD,
        "total_cases": total,
        "skipped": skipped,
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "correct": correct,
        "per_class": per_class,
        "confusion": {g: dict(confusion[g]) for g in confusion},
    }


# ── CLI ──
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Majority Vote Baseline 聚合")
    parser.add_argument("--results-dir", default="/data/eus/results/smt_v1")
    parser.add_argument("--cases-dir", default="/data/eus/datasets/smt_v1/cases")
    parser.add_argument("--output", default="/data/eus/results/smt_v1_aggregated.json")
    parser.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()
    
    print(f"Majority Vote Baseline (conf_threshold={args.conf_threshold})")
    print("=" * 60)
    
    # 聚合（使用 CLI 传入的阈值）
    aggregated = run_majority_vote_batch(args.results_dir, args.output, conf_threshold=args.conf_threshold)
    
    # 评估
    metrics = evaluate_against_ground_truth(aggregated, args.cases_dir)
    
    print(f"\n📊 评估结果:")
    print(f"  可评估病例: {metrics['total_cases']} (跳过: {metrics['skipped']})")
    print(f"  整体准确率: {metrics['correct']}/{metrics['total_cases']} = {metrics['accuracy']*100:.1f}%")
    
    print(f"\n{'类别':<8} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>7}")
    print("-" * 40)
    for cls in ["LM", "GIST", "NET", "EP", "LIP"]:
        m = metrics["per_class"][cls]
        print(f"{cls:<8} {m['precision']*100:>5.1f}% {m['recall']*100:>5.1f}% {m['f1']*100:>5.1f}% {m['support']:>7}")
    
    # 保存评估报告
    report_path = args.output.replace(".json", "_metrics.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 评估报告: {report_path}")
