"""
Evidence Chain Tracker — 证据链追踪

Multimodal Evidence Alignment (MEA) 的审计层

功能：
- 记录病例级决策的完整证据链
- 追踪每个决策步骤的输入/输出/不确定性
- 支持事后审计和复盘
- 为 LLM 进化引擎提供决策分析数据

数据结构：
- EvidenceStep: 单个证据步骤
- EvidenceChain: 病例级证据链
- AuditTrail: 管理器
"""

import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class EvidenceStep:
    """单个证据步骤"""
    step_id: str                        # 步骤 ID
    step_type: str                      # 步骤类型：feature_acquire / evaluate / decision / interaction
    timestamp: str                      # 时间戳
    
    # 输入
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # 输出
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    # 不确定性
    uncertainty: Optional[float] = None  # 0.0 ~ 1.0
    
    # 说明
    description: str = ""
    
    # 来源
    source: str = ""  # "ai" / "doctor" / "device" / "visual"


@dataclass
class EvidenceChain:
    """病例级证据链"""
    chain_id: str                       # 链 ID
    case_id: str                        # 病例 ID
    created_at: str                     # 创建时间
    
    # 步骤列表
    steps: List[EvidenceStep] = field(default_factory=list)
    
    # 最终决策
    final_decision: Optional[Dict] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: EvidenceStep):
        """添加步骤"""
        self.steps.append(step)
    
    def get_steps_by_type(self, step_type: str) -> List[EvidenceStep]:
        """按类型获取步骤"""
        return [s for s in self.steps if s.step_type == step_type]
    
    def get_last_step(self) -> Optional[EvidenceStep]:
        """获取最后一步"""
        return self.steps[-1] if self.steps else None
    
    def total_uncertainty(self) -> float:
        """计算总不确定性（各步骤不确定性累积）"""
        uncertainties = [s.uncertainty for s in self.steps if s.uncertainty is not None]
        if not uncertainties:
            return 0.0
        # 使用累积公式：1 - ∏(1 - u_i)
        product = 1.0
        for u in uncertainties:
            product *= (1 - u)
        return 1 - product


class AuditTrail:
    """证据链追踪管理器"""
    
    def __init__(self, storage_path: str = None):
        """
        Args:
            storage_path: 存储路径（JSON 文件）
        """
        if storage_path is None:
            storage_path = str(Path(__file__).parent / "evidence_chains.json")
        self.storage_path = Path(storage_path)
        self.chains: Dict[str, EvidenceChain] = {}
        self._load()
    
    def _load(self):
        """从文件加载"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # 转换 steps
                    steps = [EvidenceStep(**s) for s in item.get('steps', [])]
                    item['steps'] = steps
                    chain = EvidenceChain(**item)
                    self.chains[chain.chain_id] = chain
    
    def _save(self):
        """保存到文件"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(chain) for chain in self.chains.values()]
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create_chain(self, case_id: str, metadata: Dict = None) -> EvidenceChain:
        """创建新的证据链"""
        chain = EvidenceChain(
            chain_id=str(uuid.uuid4())[:8],
            case_id=case_id,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self.chains[chain.chain_id] = chain
        self._save()
        return chain
    
    def get_chain(self, chain_id: str) -> Optional[EvidenceChain]:
        """获取证据链"""
        return self.chains.get(chain_id)
    
    def get_chains_by_case(self, case_id: str) -> List[EvidenceChain]:
        """按病例获取证据链"""
        return [c for c in self.chains.values() if c.case_id == case_id]
    
    def add_feature_acquire_step(self,
                                  chain: EvidenceChain,
                                  feature_name: str,
                                  feature_value: Any,
                                  source: str,
                                  uncertainty: float = None) -> EvidenceStep:
        """记录特征获取步骤"""
        step = EvidenceStep(
            step_id=str(uuid.uuid4())[:8],
            step_type="feature_acquire",
            timestamp=datetime.now().isoformat(),
            inputs={"feature_name": feature_name},
            outputs={"feature_value": feature_value},
            uncertainty=uncertainty,
            description=f"获取特征 {feature_name} = {feature_value}",
            source=source,
        )
        chain.add_step(step)
        self._save()
        return step
    
    def add_evaluate_step(self,
                          chain: EvidenceChain,
                          predictions: Dict[str, float],
                          consistency: float,
                          available_features: Dict[str, Any],
                          sufficiency: float,
                          action: str) -> EvidenceStep:
        """记录评估步骤"""
        step = EvidenceStep(
            step_id=str(uuid.uuid4())[:8],
            step_type="evaluate",
            timestamp=datetime.now().isoformat(),
            inputs={
                "predictions": predictions,
                "consistency": consistency,
                "available_features": available_features,
            },
            outputs={
                "sufficiency": sufficiency,
                "action": action,
            },
            uncertainty=1 - sufficiency,  # 充分性的反面是不确定性
            description=f"评估: 充分性={sufficiency:.2f}, 行动={action}",
            source="ai",
        )
        chain.add_step(step)
        self._save()
        return step
    
    def add_decision_step(self,
                          chain: EvidenceChain,
                          diagnosis: str,
                          confidence: float,
                          candidates: List[Dict],
                          supporting_evidence: List[str],
                          opposing_evidence: List[str]) -> EvidenceStep:
        """记录决策步骤"""
        step = EvidenceStep(
            step_id=str(uuid.uuid4())[:8],
            step_type="decision",
            timestamp=datetime.now().isoformat(),
            inputs={
                "candidates": candidates,
            },
            outputs={
                "diagnosis": diagnosis,
                "confidence": confidence,
                "supporting_evidence": supporting_evidence,
                "opposing_evidence": opposing_evidence,
            },
            uncertainty=1 - confidence,
            description=f"决策: {diagnosis} ({confidence:.1%})",
            source="ai",
        )
        chain.add_step(step)
        
        # 更新最终决策
        chain.final_decision = {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "candidates": candidates,
            "supporting_evidence": supporting_evidence,
            "opposing_evidence": opposing_evidence,
        }
        self._save()
        return step
    
    def add_interaction_step(self,
                             chain: EvidenceChain,
                             interaction_type: str,
                             request: Dict,
                             response: Dict = None) -> EvidenceStep:
        """记录交互步骤"""
        step = EvidenceStep(
            step_id=str(uuid.uuid4())[:8],
            step_type="interaction",
            timestamp=datetime.now().isoformat(),
            inputs={"request": request},
            outputs={"response": response} if response else {},
            description=f"交互: {interaction_type}",
            source="doctor" if response else "ai",
        )
        chain.add_step(step)
        self._save()
        return step
    
    def add_correction_step(self,
                            chain: EvidenceChain,
                            feedback_id: str,
                            correct_diagnosis: str,
                            error_type: str) -> EvidenceStep:
        """记录纠正步骤"""
        step = EvidenceStep(
            step_id=str(uuid.uuid4())[:8],
            step_type="correction",
            timestamp=datetime.now().isoformat(),
            inputs={
                "feedback_id": feedback_id,
                "original_diagnosis": chain.final_decision.get("diagnosis") if chain.final_decision else None,
            },
            outputs={
                "correct_diagnosis": correct_diagnosis,
                "error_type": error_type,
            },
            description=f"纠正: {correct_diagnosis} (错误类型: {error_type})",
            source="doctor",
        )
        chain.add_step(step)
        self._save()
        return step
    
    def export_for_audit(self, chain_id: str) -> Dict:
        """导出审计格式"""
        chain = self.get_chain(chain_id)
        if not chain:
            return {}
        
        return {
            "chain_id": chain.chain_id,
            "case_id": chain.case_id,
            "created_at": chain.created_at,
            "total_steps": len(chain.steps),
            "total_uncertainty": chain.total_uncertainty(),
            "final_decision": chain.final_decision,
            "steps": [
                {
                    "step_id": s.step_id,
                    "type": s.step_type,
                    "timestamp": s.timestamp,
                    "description": s.description,
                    "source": s.source,
                    "uncertainty": s.uncertainty,
                    "inputs": s.inputs,
                    "outputs": s.outputs,
                }
                for s in chain.steps
            ],
            "metadata": chain.metadata,
        }
    
    def export_for_llm(self, n: int = None) -> List[Dict]:
        """导出供 LLM 分析的格式"""
        chains = list(self.chains.values())
        if n:
            chains = sorted(chains, key=lambda c: c.created_at, reverse=True)[:n]
        
        return [
            {
                "case_id": c.case_id,
                "final_decision": c.final_decision,
                "total_uncertainty": c.total_uncertainty(),
                "step_types": [s.step_type for s in c.steps],
                "has_correction": any(s.step_type == "correction" for s in c.steps),
                "feature_sources": list(set(s.source for s in c.steps if s.step_type == "feature_acquire")),
            }
            for c in chains
        ]
    
    def statistics(self) -> Dict:
        """统计信息"""
        if not self.chains:
            return {"total_chains": 0}
        
        total_chains = len(self.chains)
        total_steps = sum(len(c.steps) for c in self.chains.values())
        
        # 步骤类型分布
        step_types = {}
        for c in self.chains.values():
            for s in c.steps:
                step_types[s.step_type] = step_types.get(s.step_type, 0) + 1
        
        # 纠正率
        corrected = sum(1 for c in self.chains.values() 
                       if any(s.step_type == "correction" for s in c.steps))
        
        # 平均不确定性
        uncertainties = [c.total_uncertainty() for c in self.chains.values()]
        avg_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0
        
        return {
            "total_chains": total_chains,
            "total_steps": total_steps,
            "avg_steps_per_chain": total_steps / total_chains if total_chains else 0,
            "step_type_distribution": step_types,
            "correction_rate": corrected / total_chains if total_chains else 0,
            "avg_uncertainty": avg_uncertainty,
        }
    
    def to_summary(self) -> str:
        """生成摘要"""
        stats = self.statistics()
        if stats["total_chains"] == 0:
            return "Evidence Chain Tracker: 空（暂无证据链）"
        
        lines = [
            f"Evidence Chain Tracker 摘要:",
            f"  证据链总数: {stats['total_chains']}",
            f"  步骤总数: {stats['total_steps']}",
            f"  平均步骤/链: {stats['avg_steps_per_chain']:.1f}",
            f"  纠正率: {stats['correction_rate']:.1%}",
            f"  平均不确定性: {stats['avg_uncertainty']:.3f}",
            f"",
            f"  步骤类型分布:",
        ]
        for stype, count in sorted(stats.get("step_type_distribution", {}).items(),
                                    key=lambda x: x[1], reverse=True):
            lines.append(f"    {stype}: {count}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    tracker = AuditTrail("/tmp/test_evidence_chains.json")
    
    # 创建一个完整的证据链
    chain = tracker.create_chain(case_id="CASE-SMT1_P0")
    
    # 1. 特征获取
    tracker.add_feature_acquire_step(chain, "lesion_location", "胃底", source="device")
    tracker.add_feature_acquire_step(chain, "layer_origin", "muscularis_propria", source="doctor", uncertainty=0.1)
    
    # 2. 评估
    tracker.add_evaluate_step(
        chain,
        predictions={"GIST": 0.62, "LM": 0.28, "EP": 0.10},
        consistency=0.65,
        available_features={"lesion_location": "胃底", "layer_origin": "muscularis_propria"},
        sufficiency=0.55,
        action="diagnose_with_suggestion",
    )
    
    # 3. 决策
    tracker.add_decision_step(
        chain,
        diagnosis="GIST",
        confidence=0.62,
        candidates=[
            {"diagnosis": "GIST", "confidence": 0.62},
            {"diagnosis": "LM", "confidence": 0.28},
        ],
        supporting_evidence=["胃底(P=0.75)", "固有肌层(P=0.65)"],
        opposing_evidence=["一致性不足(0.65)"],
    )
    
    # 4. 交互
    tracker.add_interaction_step(
        chain,
        interaction_type="diagnosis_with_suggestion",
        request={"suggestions": ["echo_pattern", "homogeneous"]},
    )
    
    # 5. 纠正
    tracker.add_correction_step(
        chain,
        feedback_id="fb_001",
        correct_diagnosis="LM",
        error_type="feature_error",
    )
    
    # 输出
    print(tracker.to_summary())
    print("\n--- 审计格式 ---")
    audit = tracker.export_for_audit(chain.chain_id)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
