"""
Structured Clinical Report Generator for EUS-SMT Diagnosis.

Converts frame-level AI predictions + clinical features + RAG evidence chain
into a structured, clinically-standardized EUS report format.

Report structure follows WSG (World Society of Gastroenterology) EUS reporting guidelines.
"""
import json, os
from collections import Counter
from typing import Dict, List, Any, Optional


# ──────────────────────────────────────────────
# Structured Report Schema
# ──────────────────────────────────────────────

REPORT_TEMPLATE = {
    "report_metadata": {
        "report_type": "EUS-SMT Structured Clinical Report",
        "version": "1.0",
    },
    "case_identification": {
        "case_id": "",
    },
    "ai_detection_summary": {
        "total_frames": 0,
        "frames_with_detection": 0,
        "frame_vote": "",
        "frame_agreement": 0.0,  # proportion of frames agreeing with majority
        "mean_confidence": 0.0,
        "class_distribution": {},  # {class: count}
    },
    "clinical_characteristics": {
        "location": "",
        "echo_pattern": "",
        "layer_origin": "",
        "border": "",
        "shape": "",
        "size": "",
    },
    "evidence_assessment": {
        "ai_clinical_alignment": "",  # "aligned" | "partial" | "conflict"
        "rag_support_level": "",  # "strong" | "moderate" | "weak" | "none"
        "key_supporting_evidence": [],
        "key_conflicting_evidence": [],
    },
    "diagnostic_conclusion": {
        "primary_diagnosis": "",
        "confidence_level": "",  # "definitive" | "highly_suspected" | "suspected" | "uncertain" | "indeterminate"
        "confidence_score": 0.0,
        "differential_diagnoses": [],
        "needs_review": False,
    },
    "recommendations": {
        "ai_suggestion": "",
        "action": "",
    },
    "evidence_chain_reference": {
        "reasoning": "",
        "supporting_evidence": [],
        "opposing_evidence": [],
    },
}


# ──────────────────────────────────────────────
# Clinical Knowledge Base
# ──────────────────────────────────────────────

DISEASE_PROFILES = {
    "LM": {
        "name_cn": "平滑肌瘤",
        "name_en": "Leiomyoma",
        "typical_location": ["食管(55%)", "胃底/体(45%)"],
        "typical_layer": ["固有肌层(46%)", "粘膜肌层(44%)"],
        "typical_echo": "低回声",
        "homogeneous": "93%",
        "typical_border": "清楚",
        "typical_shape": "规则",
        "recommendation": "良性肿瘤，<2cm无症状随访，>2cm或有症状考虑EMR/ESD切除",
    },
    "GIST": {
        "name_cn": "胃肠道间质瘤",
        "name_en": "Gastrointestinal Stromal Tumor",
        "typical_location": ["胃底(67%)", "胃体(25%)"],
        "typical_layer": ["固有肌层(91%)"],
        "typical_echo": "低回声",
        "homogeneous": "50%",
        "typical_border": "清楚或不规则",
        "typical_shape": "规则或分叶状",
        "recommendation": "有恶性潜能，需EUS-FNA明确危险分层，<2cm低风险可随访，>2cm考虑手术",
    },
    "NET": {
        "name_cn": "神经内分泌肿瘤",
        "name_en": "Neuroendocrine Tumor",
        "typical_location": ["直肠(94%)"],
        "typical_layer": ["粘膜肌层(48%)", "粘膜下层(45%)"],
        "typical_echo": "低回声",
        "homogeneous": "76%",
        "typical_border": "清楚",
        "typical_shape": "规则",
        "recommendation": "低度恶性潜能，<1cm可EMR/ESD切除，需Ki-67评估分级",
    },
    "EP": {
        "name_cn": "异位胰腺",
        "name_en": "Ectopic Pancreas",
        "typical_location": ["胃窦(69%)", "胃体(23%)"],
        "typical_layer": ["粘膜下层(54%)", "固有肌层(14%)"],
        "typical_echo": "低回声(74%)，可混合/高回声",
        "homogeneous": "17%",
        "typical_border": "清楚",
        "typical_shape": "规则",
        "recommendation": "良性病变，无症状无需处理，有症状或诊断不明确可ESD切除确诊",
    },
    "LIP": {
        "name_cn": "脂肪瘤",
        "name_en": "Lipoma",
        "typical_location": ["胃窦(61%)", "胃体(22%)"],
        "typical_layer": ["粘膜下层(95%)"],
        "typical_echo": "高回声(95%)",
        "homogeneous": "95%",
        "typical_border": "清楚",
        "typical_shape": "规则",
        "recommendation": "良性病变，无症状无需处理，有症状可ESD切除",
    },
}


def _compute_agreement(vote_counts: Dict[str, int]) -> float:
    """Compute frame-level agreement proportion for the majority class."""
    if not vote_counts:
        return 0.0
    total = sum(vote_counts.values())
    top = max(vote_counts.values())
    return top / total if total > 0 else 0.0


def _classify_confidence(score: float, is_uncertain: bool) -> str:
    """Map numeric confidence to clinical confidence level."""
    if is_uncertain:
        return "indeterminate"
    if score >= 0.9:
        return "definitive"
    if score >= 0.75:
        return "highly_suspected"
    if score >= 0.6:
        return "suspected"
    return "uncertain"


def _assess_ai_clinical_alignment(
    vote_pred: str,
    features: Dict[str, str],
) -> str:
    """Check if AI prediction aligns with typical clinical features."""
    if vote_pred not in DISEASE_PROFILES:
        return "unknown"
    
    profile = DISEASE_PROFILES[vote_pred]
    conflicts = 0
    
    # Check location
    loc = features.get("lesion_location", features.get("location", ""))
    if loc:
        loc_match = any(loc[:2] in t for t in profile["typical_location"])
        if not loc_match:
            conflicts += 1
    
    # Check echo
    echo = features.get("echo_pattern", "")
    if echo and "echo" in profile:
        echo_keywords = {"低": "低回声", "高": "高回声", "混合": "混合回声"}
        expected = profile["typical_echo"]
        if echo != expected[:2]:
            conflicts += 1
    
    # Check layer
    layer = features.get("layer_origin", "")
    if layer:
        layer_match = any(l[:2] in layer for l in profile["typical_layer"])
        if not layer_match:
            conflicts += 1
    
    if conflicts == 0:
        return "aligned"
    elif conflicts == 1:
        return "partial"
    else:
        return "conflict"


def _generate_recommendation(
    primary_diagnosis: str,
    confidence_level: str,
    needs_review: bool,
    features: Dict[str, str],
) -> Dict[str, str]:
    """Generate clinical recommendation based on diagnosis and evidence strength."""
    if needs_review or confidence_level in ("uncertain", "indeterminate"):
        return {
            "ai_suggestion": "病例不确定性较高",
            "action": "建议专家复核，必要时行EUS-FNA或进一步检查明确诊断",
        }
    
    if primary_diagnosis in DISEASE_PROFILES:
        profile = DISEASE_PROFILES[primary_diagnosis]
        return {
            "ai_suggestion": f"AI诊断高度提示{profile['name_cn']}",
            "action": f"{profile['recommendation']}",
        }
    
    return {
        "ai_suggestion": "AI诊断结果",
        "action": "建议结合临床资料综合判断",
    }


def _extract_vote_pred(raw_pred: str) -> str:
    """Extract standardized class from pred field."""
    valid = {"GIST", "LM", "NET", "EP", "LIP"}
    if raw_pred in valid:
        return raw_pred
    return "?"


# ──────────────────────────────────────────────
# Report Generator
# ──────────────────────────────────────────────

def generate_report(
    case_id: str,
    predictions: List[Dict],
    features: Dict[str, str],
    llm_pred: str,
    llm_reasoning: str,
    is_uncertain: bool,
    needs_review: bool,
    confidence: float,
    rag_entry: Optional[Dict] = None,
) -> Dict:
    """Generate a structured clinical report from evidence chain."""
    
    # ── AI Detection Summary ──
    total_frames = len(predictions)
    class_counts = Counter(p.get('class', '?') for p in predictions)
    vote_pred = class_counts.most_common(1)[0][0] if class_counts else '?'
    vote_pred_std = _extract_vote_pred(vote_pred)
    agreement = _compute_agreement(class_counts)
    mean_conf = sum(p.get('conf', 0) for p in predictions) / max(total_frames, 1)
    
    # ── Clinical Assessment ──
    alignment = _assess_ai_clinical_alignment(vote_pred_std, features)
    
    # ── RAG Evidence Level ──
    rag_support = "none"
    if rag_entry and rag_entry.get('reasoning'):
        rag_support = "strong"
    
    # ── Confidence Classification ──
    confidence_level = _classify_confidence(confidence, is_uncertain)
    
    # ── Differential Diagnoses ──
    differentials = []
    if rag_entry and rag_entry.get('differential'):
        for diff in rag_entry['differential']:
            if isinstance(diff, dict):
                differentials.append({
                    'diagnosis': diff.get('diagnosis', ''),
                    'probability': diff.get('probability', 0),
                    'reason': diff.get('reason', ''),
                })
            elif isinstance(diff, str):
                differentials.append({'diagnosis': diff, 'reason': ''})
    
    # If uncertain and no differentials, generate based on confusion profiles
    if not differentials and (is_uncertain or confidence < 0.7):
        if llm_pred in {"GIST", "LM", "NET", "EP", "LIP"}:
            # Generate relevant differentials based on typical confusion pairs
            confusion_map = {
                "LM": [("GIST", 0.3), ("EP", 0.05)],
                "GIST": [("LM", 0.3), ("NET", 0.05)],
                "NET": [("EP", 0.2), ("GIST", 0.1)],
                "EP": [("GIST", 0.3), ("LM", 0.2)],
                "LIP": [("EP", 0.1)],
            }
            for diag, prob in confusion_map.get(llm_pred, []):
                differentials.append({
                    'diagnosis': f'{diag} ({DISEASE_PROFILES.get(diag, {}).get("name_cn", diag)})',
                    'probability': prob,
                    'reason': f'需排除{diag}的鉴别诊断',
                })
    
    # ── Recommendations ──
    recommendations = _generate_recommendation(
        llm_pred if not is_uncertain else '',
        confidence_level,
        needs_review,
        features,
    )
    
    # ── Evidence Chain ──
    supporting = []
    opposing = []
    if rag_entry:
        if rag_entry.get('supporting_evidence'):
            supporting = rag_entry['supporting_evidence'] if isinstance(rag_entry['supporting_evidence'], list) else [rag_entry['supporting_evidence']]
        if rag_entry.get('opposing_evidence'):
            opposing = rag_entry['opposing_evidence'] if isinstance(rag_entry['opposing_evidence'], list) else [rag_entry['opposing_evidence']]
    
    # ── Build Report ──
    report = {
        "report_metadata": {
            "report_type": "EUS-SMT Structured Clinical Report",
            "version": "1.0",
        },
        "case_identification": {
            "case_id": case_id,
        },
        "ai_detection_summary": {
            "total_frames": total_frames,
            "frames_with_detection": total_frames,
            "frame_vote": vote_pred_std,
            "frame_agreement": round(agreement, 3),
            "mean_confidence": round(mean_conf, 3),
            "class_distribution": dict(class_counts),
        },
        "clinical_characteristics": {
            "location": features.get("lesion_location", features.get("location", "")),
            "echo_pattern": features.get("echo_pattern", ""),
            "layer_origin": features.get("layer_origin", ""),
            "border": features.get("border", features.get("layer_confidence", "")),
            "shape": features.get("shape", ""),
            "size": features.get("tumor_size", features.get("size", "")),
        },
        "evidence_assessment": {
            "ai_clinical_alignment": alignment,
            "rag_support_level": rag_support,
            "key_supporting_evidence": supporting[:3],
            "key_conflicting_evidence": opposing[:3],
        },
        "diagnostic_conclusion": {
            "primary_diagnosis": llm_pred if not is_uncertain else "不确定",
            "confidence_level": confidence_level,
            "confidence_score": round(confidence, 2),
            "differential_diagnoses": differentials[:3],
            "needs_review": needs_review or is_uncertain,
            "original_reasoning": llm_reasoning,
        },
        "recommendations": recommendations,
    }
    
    return report


# ──────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────

def evaluate_report(report: Dict, gt_label: str) -> Dict:
    """Evaluate a structured clinical report against ground truth."""
    metrics = {}
    
    # 1. Diagnostic Accuracy
    pred = report["diagnostic_conclusion"]["primary_diagnosis"]
    metrics["diagnosis_correct"] = (pred == gt_label)
    metrics["diagnosis_abstained"] = (pred == "不确定")
    
    # 2. Confidence Calibration
    # Definitive/highly_suspected should be correct; uncertain should abstain
    conf_level = report["diagnostic_conclusion"]["confidence_level"]
    if conf_level in ("definitive", "highly_suspected"):
        metrics["confidence_calibrated"] = (pred == gt_label)
    elif conf_level in ("uncertain", "indeterminate"):
        metrics["confidence_calibrated"] = (pred == "不确定")
    else:
        metrics["confidence_calibrated"] = None  # borderline cases
    
    # 3. Differential Diagnosis Quality
    diffs = report["diagnostic_conclusion"]["differential_diagnoses"]
    metrics["has_differential"] = len(diffs) > 0
    metrics["n_differentials"] = len(diffs)
    metrics["gt_in_differential"] = any(gt_label in d.get("diagnosis", "") for d in diffs)
    
    # 4. Evidence Completeness
    evidence = report["evidence_assessment"]
    metrics["alignment_assessed"] = evidence["ai_clinical_alignment"] != ""
    metrics["has_supporting_evidence"] = len(evidence["key_supporting_evidence"]) > 0
    metrics["has_conflicting_evidence"] = len(evidence["key_conflicting_evidence"]) > 0
    metrics["rag_support_available"] = evidence["rag_support_level"] != "none"
    
    # 5. Recommendation Completeness
    rec = report["recommendations"]
    metrics["has_recommendation"] = bool(rec.get("action", ""))
    metrics["has_ai_suggestion"] = bool(rec.get("ai_suggestion", ""))
    
    # 6. Report Completeness Score (0-100)
    completeness_score = 0
    checks = [
        bool(report["clinical_characteristics"]["location"]),
        bool(report["clinical_characteristics"]["echo_pattern"]),
        bool(report["clinical_characteristics"]["layer_origin"]),
        metrics["has_differential"],
        metrics["has_recommendation"],
        metrics["alignment_assessed"],
    ]
    completeness_score = sum(1 for c in checks if c) / len(checks) * 100
    metrics["completeness_score"] = completeness_score
    
    return metrics


def generate_and_evaluate_all(
    results_data: Dict,
    rag_index: List[Dict],
    gt_labels: Dict[str, str],
    predictions_cache: Dict[str, List],
    features_cache: Dict[str, Dict],
) -> Dict:
    """Generate reports for all cases and compute aggregate metrics."""
    
    # Build RAG index lookup
    rag_lookup = {e['case_id']: e for e in rag_index}
    
    all_reports = []
    all_metrics = []
    
    for k_key, k_data in results_data.items():
        if k_key.startswith('k_'):
            for bk in ['batch_0', 'batch_1', 'batch_2', 'batch_3']:
                if bk in k_data:
                    for r in k_data[bk]['results']:
                        cid = r['case_id']
                        report = generate_report(
                            case_id=cid,
                            predictions=predictions_cache.get(cid, []),
                            features=features_cache.get(cid, {}),
                            llm_pred=r.get('pred', '?'),
                            llm_reasoning=r.get('reasoning', ''),
                            is_uncertain=r.get('is_uncertain', False),
                            needs_review=r.get('needs_review', False),
                            confidence=r.get('confidence', 0.5),
                            rag_entry=rag_lookup.get(cid),
                        )
                        metrics = evaluate_report(report, gt_labels.get(cid, ''))
                        all_reports.append(report)
                        all_metrics.append(metrics)
    
    # Aggregate metrics
    n = len(all_metrics)
    aggregated = {
        "n_reports": n,
        "diagnosis_accuracy": sum(1 for m in all_metrics if m.get('diagnosis_correct')) / n,
        "abstention_rate": sum(1 for m in all_metrics if m.get('diagnosis_abstained')) / n,
        "mean_completeness": sum(m.get('completeness_score', 0) for m in all_metrics) / n,
        "differential_presence_rate": sum(1 for m in all_metrics if m.get('has_differential')) / n,
        "gt_in_differential_rate": sum(1 for m in all_metrics if m.get('gt_in_differential')) / max(n, 1),
        "supporting_evidence_rate": sum(1 for m in all_metrics if m.get('has_supporting_evidence')) / n,
        "conflicting_evidence_rate": sum(1 for m in all_metrics if m.get('has_conflicting_evidence')) / n,
        "recommendation_rate": sum(1 for m in all_metrics if m.get('has_recommendation')) / n,
        "alignment_assessed_rate": sum(1 for m in all_metrics if m.get('alignment_assessed')) / n,
    }
    
    return {
        "aggregate_metrics": aggregated,
        "reports": all_reports,
        "per_case_metrics": all_metrics,
    }
