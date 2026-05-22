"""
LLM Evidence Evaluator — LLM 驱动的证据评估器

替代原 evidence_evaluator.py 中的规则逻辑，用 LLM 进行证据推理。

用法：
    from harness.llm import LLMClient, PromptTemplates, SchemaValidator, UncertaintyGate
    from harness.reasoners.evidence_chain import DiagnosticReasoner
    
    llm = LLMClient(model="qwen")
    evaluator = DiagnosticReasoner(llm)
    
    result = evaluator.evaluate(
        case_id="CASE-001",
        predictions=[{"frame_id": "f1", "class": "GIST", "conf": 0.8}, ...],
        features={"lesion_location": "胃底", "echo_pattern": "低回声"},
    )
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from harness.llm_client import LLMClient
from harness.reasoners.prompts import PromptTemplates
from harness.validators.schema import SchemaValidator
from harness.validators.gate import AgreementValidator

logger = logging.getLogger(__name__)


@dataclass
class LLMEvaluationResult:
    """LLM 评估结果"""
    # 最终决策
    action: str                          # diagnose / differential / request_review
    diagnosis: str                       # 最终诊断
    confidence: float                    # 置信度
    
    # 证据
    supporting_evidence: List[Dict]      # 支持证据
    opposing_evidence: List[Dict]        # 反对证据
    differential: List[Dict]             # 鉴别诊断
    uncertainty_sources: List[str]       # 不确定性来源
    
    # 充分性
    sufficiency_score: float
    sufficiency_level: str
    missing_features: List[str]
    recommendation: str
    
    # LLM 推理过程
    reasoning: str                       # LLM 的推理过程
    
    # 门控信息
    gate_passed: bool
    gate_warnings: List[str]
    gate_overrides: List[str]
    
    # 元数据
    latency_ms: float = 0.0
    model: str = ""
    token_usage: Dict = field(default_factory=dict)


class DiagnosticReasoner:
    """LLM 驱动的证据评估器"""
    
    def __init__(self, 
                 llm: LLMClient):
        """
        Args:
            llm: LLM 客户端
        """
        self.llm = llm
        self.agreement = AgreementValidator()
        self.parser = SchemaValidator()
    
    def evaluate(self,
                 case_id: str,
                 predictions: List[Dict],
                 features: Dict[str, str],
                 ground_truth: str = None) -> LLMEvaluationResult:
        """
        评估一个病例的证据

        Args:
            case_id: 病例 ID
            predictions: 帧级预测 [{"frame_id": ..., "class": ..., "conf": ...}, ...]
            features: 临床特征 {"lesion_location": "胃底", ...}
            ground_truth: 真实标签（仅用于实验记录，不影响评估）

        Returns:
            LLMEvaluationResult
        """
        import time
        from collections import Counter
        start = time.time()

        # 0. 计算 detector 多数投票
        vote_classes = [p['class'] for p in predictions]
        vote_counter = Counter(vote_classes)
        detector_vote = vote_counter.most_common(1)[0][0] if vote_classes else '?'

        # 1. 构造 prompt
        messages = PromptTemplates.build_evidence_chain_prompt(
            case_id=case_id,
            predictions=predictions,
            features=features,
            ground_truth=ground_truth,
        )

        # 2. 调用 LLM
        try:
            raw_output = self.llm.chat_json(
                messages,
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return self._fallback_result(case_id, str(e))

        latency = (time.time() - start) * 1000

        # 3. 解析 + Schema 验证
        parse_result = self.parser.parse_evidence_chain(raw_output)

        if not parse_result.success:
            logger.warning(f"输出解析失败: {parse_result.errors}")
            return self._fallback_result(case_id, f"解析失败: {parse_result.errors}")

        data = parse_result.data

        # 4. LLM-Detector Agreement 验证
        llm_diagnosis = data.get("diagnosis", "uncertain")
        confidence = data.get("confidence", 0.0)
        gate_decision = self.agreement.validate(llm_diagnosis, detector_vote, confidence)

        # 5. 组装结果
        sufficiency = data.get("sufficiency", {})

        return LLMEvaluationResult(
            action=gate_decision.action,
            diagnosis=llm_diagnosis if gate_decision.passed else "uncertain",
            confidence=gate_decision.confidence_adjusted,
            supporting_evidence=data.get("supporting_evidence", []),
            opposing_evidence=data.get("opposing_evidence", []),
            differential=data.get("differential", []),
            uncertainty_sources=data.get("uncertainty_sources", []),
            sufficiency_score=sufficiency.get("score", 0.5),
            sufficiency_level=sufficiency.get("level", "medium"),
            missing_features=sufficiency.get("missing_features", []),
            recommendation=sufficiency.get("recommendation", ""),
            reasoning=data.get("reasoning", ""),
            gate_passed=gate_decision.passed,
            gate_warnings=gate_decision.warnings,
            gate_overrides=[],
            latency_ms=latency,
            model=self.llm.model_name,
            token_usage=raw_output.get("usage", {}) if isinstance(raw_output, dict) else {},
        )
    
    def _fallback_result(self, case_id: str, error: str) -> LLMEvaluationResult:
        """LLM 调用失败时的兜底结果"""
        return LLMEvaluationResult(
            action="request_review",
            diagnosis="uncertain",
            confidence=0.0,
            supporting_evidence=[],
            opposing_evidence=[],
            differential=[],
            uncertainty_sources=[f"系统错误: {error}"],
            sufficiency_score=0.0,
            sufficiency_level="low",
            missing_features=[],
            recommendation="系统异常，建议人工复核",
            reasoning=f"LLM 调用失败: {error}",
            gate_passed=False,
            gate_warnings=[f"系统错误: {error}"],
            gate_overrides=["error → request_review"],
        )
    
    def to_dict(self, result: LLMEvaluationResult) -> Dict:
        """转换为字典（便于序列化）"""
        return {
            "action": result.action,
            "diagnosis": result.diagnosis,
            "confidence": result.confidence,
            "supporting_evidence": result.supporting_evidence,
            "opposing_evidence": result.opposing_evidence,
            "differential": result.differential,
            "uncertainty_sources": result.uncertainty_sources,
            "sufficiency": {
                "score": result.sufficiency_score,
                "level": result.sufficiency_level,
                "missing_features": result.missing_features,
                "recommendation": result.recommendation,
            },
            "reasoning": result.reasoning,
            "gate": {
                "passed": result.gate_passed,
                "warnings": result.gate_warnings,
                "overrides": result.gate_overrides,
            },
            "metadata": {
                "latency_ms": result.latency_ms,
                "model": result.model,
                "token_usage": result.token_usage,
            },
        }
