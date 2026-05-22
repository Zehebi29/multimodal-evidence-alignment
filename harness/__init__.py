from __future__ import annotations
"""
Multimodal Evidence Alignment (MEA)

Identifies cases where frame-level AI predictions are likely erroneous
through multimodal evidence retrieval, protocol-constrained LLM reasoning,
and cross-model agreement validation.

Organized as a four-panel evidence lifecycle DAG:

  Tools (T)      — Stateless external computation (detector, embedder, aggregator)
  Reasoners (R)  — LLM calls with typed I/O (diagnostic reasoning, evidence chain)
  Validators (V) — Check reasoner outputs (schema, agreement, uncertainty gate)
  Slots (S)      — Named evidence storage (evidence store, feedback store, audit trail)
"""

__version__ = "0.3.0"

# ── Tools ──
from harness.tools.aggregator import aggregate_majority_vote as majority_vote

# ── Reasoners ──
from harness.reasoners.evidence_chain import DiagnosticReasoner
from harness.reasoners.prompts import PromptTemplates

# ── Validators ──
from harness.validators.gate import AgreementValidator, GateDecision
from harness.validators.schema import SchemaValidator, ParseResult

# ── Slots ──
from harness.slots.evidence_store import EvidenceStore, CaseEmbedding
from harness.slots.audit_trail import AuditTrail
from harness.slots.feedback_store import FeedbackStore

# ── Registry ──
from harness.registry.feature import FeatureRegistry, Feature

# ── Clinical ──
from harness.clinical import ClinicalInteraction

# ── LLM Client ──
from harness.llm_client import LLMClient

# ── Main Harness ──
from harness.harness import MultimodalEvidenceAlignment
