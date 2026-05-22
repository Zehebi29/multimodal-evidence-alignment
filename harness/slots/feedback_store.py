"""
Feedback Store — 医生反馈收集与存储

Multimodal Evidence Alignment (MEA) 的反馈数据层

功能：
- 存储医生对 AI 诊断的纠正
- 记录证据级别的错误标注
- 支持按时间/病例/诊断查询
- 为 LLM 进化引擎提供数据源

数据结构：
- FeedbackEntry: 单条反馈记录
- EvidenceCorrection: 证据级别的纠正
- FeedbackStore: 反馈存储管理器
"""

import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class EvidenceCorrection:
    """证据级别的纠正"""
    feature_name: str           # 特征名（如 layer_origin）
    ai_value: Optional[str]     # AI 的判断值
    correct_value: str          # 医生纠正的正确值
    impact: str                 # 对诊断的影响："支持" / "反对" / "无关"
    note: Optional[str] = None  # 医生备注


@dataclass
class FeedbackEntry:
    """单条反馈记录"""
    feedback_id: str                    # 唯一 ID
    case_id: str                        # 病例 ID
    
    # AI 的原始输出
    ai_diagnosis: str                   # AI 的诊断
    ai_confidence: float                # AI 的置信度
    ai_candidates: List[Dict]           # AI 的候选列表 [{diagnosis, confidence}]
    ai_sufficiency: float               # AI 的充分性分数
    ai_action: str                      # AI 的行动（diagnose/request_info/differential）
    
    # 医生的纠正
    correct_diagnosis: str              # 正确诊断
    evidence_corrections: List[EvidenceCorrection] = field(default_factory=list)
    
    # 元数据
    doctor_id: Optional[str] = None     # 医生 ID
    confidence: str = "certain"         # 医生确信度：certain / probable / uncertain
    note: Optional[str] = None          # 医生备注
    timestamp: str = ""                 # 时间戳
    
    # 分析标签（LLM 后续标注）
    error_type: Optional[str] = None    # 错误类型：feature_error / aggregation_error / threshold_error
    severity: Optional[str] = None      # 严重程度：high / medium / low
    
    def __post_init__(self):
        if not self.feedback_id:
            self.feedback_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class FeedbackStore:
    """反馈存储管理器"""
    
    def __init__(self, storage_path: str = None):
        """
        Args:
            storage_path: 存储路径（JSON 文件）
        """
        if storage_path is None:
            storage_path = str(Path(__file__).parent / "feedback_store.json")
        self.storage_path = Path(storage_path)
        self.entries: List[FeedbackEntry] = []
        self._load()
    
    def _load(self):
        """从文件加载"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # 转换 evidence_corrections
                    corrections = [
                        EvidenceCorrection(**c) for c in item.get('evidence_corrections', [])
                    ]
                    item['evidence_corrections'] = corrections
                    self.entries.append(FeedbackEntry(**item))
    
    def _save(self):
        """保存到文件"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for entry in self.entries:
            d = asdict(entry)
            data.append(d)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add(self, entry: FeedbackEntry) -> str:
        """添加一条反馈"""
        self.entries.append(entry)
        self._save()
        return entry.feedback_id
    
    def get_by_id(self, feedback_id: str) -> Optional[FeedbackEntry]:
        """按 ID 查询"""
        for entry in self.entries:
            if entry.feedback_id == feedback_id:
                return entry
        return None
    
    def get_by_case(self, case_id: str) -> List[FeedbackEntry]:
        """按病例查询"""
        return [e for e in self.entries if e.case_id == case_id]
    
    def get_by_diagnosis(self, diagnosis: str) -> List[FeedbackEntry]:
        """按正确诊断查询"""
        return [e for e in self.entries if e.correct_diagnosis == diagnosis]
    
    def get_errors_by_type(self, error_type: str) -> List[FeedbackEntry]:
        """按错误类型查询"""
        return [e for e in self.entries if e.error_type == error_type]
    
    def get_recent(self, n: int = 10) -> List[FeedbackEntry]:
        """获取最近 n 条"""
        return sorted(self.entries, key=lambda e: e.timestamp, reverse=True)[:n]
    
    def statistics(self) -> Dict:
        """统计信息"""
        if not self.entries:
            return {"total": 0}
        
        # 错误类型统计
        error_types = {}
        for e in self.entries:
            if e.error_type:
                error_types[e.error_type] = error_types.get(e.error_type, 0) + 1
        
        # 正确诊断分布
        correct_dist = {}
        for e in self.entries:
            correct_dist[e.correct_diagnosis] = correct_dist.get(e.correct_diagnosis, 0) + 1
        
        # AI 错误诊断分布
        ai_wrong_dist = {}
        for e in self.entries:
            if e.ai_diagnosis != e.correct_diagnosis:
                ai_wrong_dist[e.ai_diagnosis] = ai_wrong_dist.get(e.ai_diagnosis, 0) + 1
        
        # 证据纠正频率
        feature_corrections = {}
        for e in self.entries:
            for c in e.evidence_corrections:
                feature_corrections[c.feature_name] = feature_corrections.get(c.feature_name, 0) + 1
        
        return {
            "total": len(self.entries),
            "error_types": error_types,
            "correct_diagnosis_distribution": correct_dist,
            "ai_wrong_diagnosis_distribution": ai_wrong_dist,
            "feature_correction_frequency": feature_corrections,
            "accuracy": 1 - len([e for e in self.entries if e.ai_diagnosis != e.correct_diagnosis]) / len(self.entries),
        }
    
    def export_for_llm(self, n: int = None) -> List[Dict]:
        """导出供 LLM 分析的格式"""
        entries = self.entries if n is None else self.get_recent(n)
        return [
            {
                "case_id": e.case_id,
                "ai_diagnosis": e.ai_diagnosis,
                "ai_confidence": e.ai_confidence,
                "correct_diagnosis": e.correct_diagnosis,
                "error_type": e.error_type,
                "evidence_corrections": [
                    {
                        "feature": c.feature_name,
                        "ai_value": c.ai_value,
                        "correct_value": c.correct_value,
                        "impact": c.impact,
                    }
                    for c in e.evidence_corrections
                ],
                "doctor_note": e.note,
            }
            for e in entries
        ]
    
    def to_summary(self) -> str:
        """生成摘要"""
        stats = self.statistics()
        if stats["total"] == 0:
            return "Feedback Store: 空（暂无反馈）"
        
        lines = [
            f"Feedback Store 摘要:",
            f"  总反馈数: {stats['total']}",
            f"  AI 准确率: {stats['accuracy']:.1%}",
            f"",
            f"  错误类型分布:",
        ]
        for etype, count in stats.get("error_types", {}).items():
            lines.append(f"    {etype}: {count}")
        
        lines.append(f"")
        lines.append(f"  证据纠正频率:")
        for feat, count in sorted(stats.get("feature_correction_frequency", {}).items(), 
                                   key=lambda x: x[1], reverse=True):
            lines.append(f"    {feat}: {count}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    store = FeedbackStore("/tmp/test_feedback_store.json")
    
    # 添加一条示例反馈
    entry = FeedbackEntry(
        feedback_id="",
        case_id="CASE-SMT1_P0",
        ai_diagnosis="GIST",
        ai_confidence=0.62,
        ai_candidates=[
            {"diagnosis": "GIST", "confidence": 0.62},
            {"diagnosis": "LM", "confidence": 0.28},
            {"diagnosis": "EP", "confidence": 0.10},
        ],
        ai_sufficiency=0.55,
        ai_action="diagnose_with_suggestion",
        correct_diagnosis="LM",
        evidence_corrections=[
            EvidenceCorrection(
                feature_name="lesion_location",
                ai_value="胃底",
                correct_value="胃底",
                impact="无关",
                note="位置正确",
            ),
            EvidenceCorrection(
                feature_name="layer_origin",
                ai_value=None,
                correct_value="submucosa",
                impact="反对",
                note="AI 未获取此特征，实际是粘膜下层",
            ),
        ],
        doctor_id="dr_zheng",
        confidence="certain",
        note="该病例为粘膜下层起源的平滑肌瘤，AI 误判为 GIST",
        error_type="feature_error",
        severity="medium",
    )
    
    store.add(entry)
    
    print(store.to_summary())
    print("\n--- 导出供 LLM 分析 ---")
    print(json.dumps(store.export_for_llm(), ensure_ascii=False, indent=2))
