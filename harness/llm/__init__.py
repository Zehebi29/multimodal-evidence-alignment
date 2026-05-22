"""
LLM 模块 — Multimodal Evidence Alignment (MEA) 的 LLM 控制层

三层控制：
- prompts: 结构化 Prompt 模板（控制输入）
- client: LLM API 调用（受控推理）
- parsers: 输出解析 + 门控（控制输出）
"""

from harness.llm_client import LLMClient
from harness.reasoners.prompts import PromptTemplates
from harness.validators.schema import SchemaValidator
from harness.validators.gate import UncertaintyGate
from harness.reasoners.evidence_chain import DiagnosticReasoner, LLMEvaluationResult
from harness.llm.evolution import LLMEvolutionEngine, EvolutionReport

__all__ = [
    "LLMClient", 
    "PromptTemplates", 
    "SchemaValidator", 
    "UncertaintyGate",
    "DiagnosticReasoner",
    "LLMEvaluationResult",
    "LLMEvolutionEngine",
    "EvolutionReport",
]
