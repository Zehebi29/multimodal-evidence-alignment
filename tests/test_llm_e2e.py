"""
端到端测试 — LLM 驱动的 Multimodal Evidence Alignment (MEA)

用 373 例 EUS-SMT 数据测试 LLM 证据链评估。
"""

import sys
import os
import json
import glob
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.llm import LLMClient, DiagnosticReasoner, UncertaintyGate

# ── 配置 ──
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-nnfrfaoikyexvlfrvwawcenqaeiqzjltvhviqttqzacwykod")
MODEL = "qwen"  # 72B 模型，质量高
RESULTS_DIR = "/data/eus/results/smt_v1/"
DATASET_DIR = "/data/eus/datasets/smt_v1/cases/"
OUTPUT_FILE = "/data/eus/results/llm_evaluation_results.json"

# 诊断映射
DIAG_MAP = {
    "平滑肌瘤": "LM", "LM": "LM", "Leiomyoma": "LM",
    "胃肠道间质瘤": "GIST", "GIST": "GIST",
    "神经内分泌肿瘤": "NET", "NET": "NET",
    "异位胰腺": "EP", "EP": "EP",
    "脂肪瘤": "LIP", "LIP": "LIP",
}

# 类名映射（处理 class index）
CLASS_MAP = {"0": "LM", "1": "NET", "2": "GIST", "3": "EP", "4": "LIP"}


def load_data():
    """加载所有数据"""
    # 1. 加载 ground truth 和临床特征
    gt_map = {}
    features_map = {}
    case_dirs = sorted(glob.glob(os.path.join(DATASET_DIR, "CASE-*")))
    
    for d in case_dirs:
        meta_path = os.path.join(d, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            m = json.load(f)
        
        case_id = os.path.basename(d)
        diag_raw = m.get("pathology_ground_truth", {}).get("diagnosis", "")
        diag = DIAG_MAP.get(diag_raw, diag_raw)
        gt_map[case_id] = diag
        
        ci = m.get("clinical_info", {})
        features_map[case_id] = {
            "lesion_location": ci.get("lesion_location", ""),
            "echo_pattern": ci.get("echo_pattern", ""),
            "layer_origin": ci.get("layer_origin", ""),
            "homogeneous": ci.get("homogeneous", ""),
        }
    
    # 2. 加载检测结果
    det_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    det_map = {}
    
    for fpath in det_files:
        with open(fpath) as f:
            case = json.load(f)
        case_id = case["case_id"]
        
        # 提取帧级预测
        predictions = []
        for frame in case.get("frames", []):
            dets = frame.get("detections", [])
            if dets:
                det = dets[0]  # 取最高置信度的检测
                cls = det.get("class_name", det.get("class", "?"))
                cls = CLASS_MAP.get(cls, cls)
                predictions.append({
                    "frame_id": frame["frame_id"],
                    "class": cls,
                    "conf": det.get("conf", 0),
                })
        
        if predictions:
            det_map[case_id] = predictions
    
    return gt_map, features_map, det_map


def run_evaluation(evaluator, cases, gt_map, features_map, det_map, 
                   max_cases=None, save_every=10):
    """运行 LLM 评估"""
    results = []
    correct = 0
    total = 0
    errors = []
    
    case_list = list(cases)
    if max_cases:
        case_list = case_list[:max_cases]
    
    print(f"\n{'='*60}")
    print(f"Running LLM evaluation on {len(case_list)} cases...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for i, case_id in enumerate(case_list):
        if case_id not in det_map:
            continue
        
        predictions = det_map[case_id]
        features = features_map.get(case_id, {})
        gt = gt_map.get(case_id, "unknown")
        
        # 跳过 "其他" 类别（无法评估）
        if gt in ("其他", "unknown"):
            continue
        
        try:
            result = evaluator.evaluate(
                case_id=case_id,
                predictions=predictions,
                features=features,
                ground_truth=gt,
            )
            
            total += 1
            is_correct = result.diagnosis == gt
            if is_correct:
                correct += 1
            
            result_entry = {
                "case_id": case_id,
                "ground_truth": gt,
                "llm_diagnosis": result.diagnosis,
                "confidence": result.confidence,
                "correct": is_correct,
                "action": result.action,
                "sufficiency": result.sufficiency_score,
                "reasoning": result.reasoning[:200],
                "supporting_count": len(result.supporting_evidence),
                "opposing_count": len(result.opposing_evidence),
                "differential_count": len(result.differential),
                "gate_passed": result.gate_passed,
                "gate_warnings": result.gate_warnings,
                "latency_ms": result.latency_ms,
            }
            results.append(result_entry)
            
            # 进度输出
            acc = correct / total if total > 0 else 0
            status = "✓" if is_correct else "✗"
            elapsed = time.time() - start_time
            avg_time = elapsed / total
            eta = avg_time * (len(case_list) - i - 1)
            
            print(f"[{i+1}/{len(case_list)}] {status} {case_id}: "
                  f"GT={gt} LLM={result.diagnosis} "
                  f"conf={result.confidence:.0%} "
                  f"acc={acc:.1%} "
                  f"({result.latency_ms:.0f}ms) "
                  f"ETA={eta/60:.1f}min")
            
            # 定期保存
            if total % save_every == 0:
                save_results(results, correct, total)
                
        except Exception as e:
            errors.append({"case_id": case_id, "error": str(e)})
            print(f"[{i+1}/{len(case_list)}] ERROR {case_id}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed/60:.1f} minutes")
    print(f"Accuracy: {correct}/{total} = {correct/total:.1%}" if total > 0 else "No valid cases")
    print(f"Errors: {len(errors)}")
    print(f"{'='*60}")
    
    save_results(results, correct, total, errors)
    return results, correct, total, errors


def save_results(results, correct, total, errors=None):
    """保存结果"""
    output = {
        "summary": {
            "total_evaluated": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "errors": len(errors) if errors else 0,
        },
        "results": results,
        "errors": errors or [],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=None, help="Max cases to evaluate")
    parser.add_argument("--model", default=MODEL, help="LLM model name")
    parser.add_argument("--save-every", type=int, default=10, help="Save every N cases")
    args = parser.parse_args()
    
    # 设置 API Key
    os.environ["SILICONFLOW_API_KEY"] = API_KEY
    
    # 加载数据
    print("Loading data...")
    gt_map, features_map, det_map = load_data()
    print(f"Loaded: {len(gt_map)} GT, {len(features_map)} features, {len(det_map)} detections")
    
    # 找到有检测结果且有 ground truth 的有效病例
    valid_cases = [cid for cid in det_map if cid in gt_map and gt_map[cid] not in ("其他", "unknown")]
    print(f"Valid cases: {len(valid_cases)}")
    
    # 初始化 LLM
    llm = LLMClient(model=args.model)
    gate = UncertaintyGate(min_confidence=0.2, high_confidence=0.6)
    evaluator = DiagnosticReasoner(llm, gate)
    
    # 运行评估
    results, correct, total, errors = run_evaluation(
        evaluator, valid_cases, gt_map, features_map, det_map,
        max_cases=args.max_cases,
        save_every=args.save_every,
    )
    
    # 输出详细统计
    if results:
        # 按类别统计
        by_class = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            gt = r["ground_truth"]
            by_class[gt]["total"] += 1
            if r["correct"]:
                by_class[gt]["correct"] += 1
        
        print("\n=== Per-class Accuracy ===")
        for cls, stats in sorted(by_class.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {cls}: {stats['correct']}/{stats['total']} = {acc:.1%}")
        
        # 门控统计
        gate_passed = sum(1 for r in results if r["gate_passed"])
        print(f"\n=== Gate Statistics ===")
        print(f"  Passed: {gate_passed}/{len(results)} = {gate_passed/len(results):.1%}")
        
        # 平均延迟
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        print(f"\n=== Performance ===")
        print(f"  Avg latency: {avg_latency:.0f}ms")


if __name__ == "__main__":
    main()
