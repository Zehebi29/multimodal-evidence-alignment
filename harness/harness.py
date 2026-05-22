"""
Multimodal Evidence Alignment (MEA) — Experimental Pipeline

The exact pipeline described in the MEA paper:
  Offline: FrameDetector → CaseAggregator → EvidenceChainGenerator(LLM + Protocol) → EvidenceStore
  Online:  FrameDetector → CaseAggregator → SimilarCaseRetriever → DiagnosticReasoner(LLM, short prompt)
           → AgreementValidator → UncertaintyGate / ReportGenerator

This file orchestrates the two-phase experimental pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .validators.gate import AgreementValidator


@dataclass
class PipelineResult:
    """Result of a single MEA inference pass."""
    case_id: str
    llm_diagnosis: str
    detector_vote: str
    uncertain: bool          # True = flagged for expert review
    confidence: float
    reasoning: str


class MultimodalEvidenceAlignment:
    """MEA experimental pipeline (matches the paper exactly)."""

    def __init__(self):
        self.validator = AgreementValidator()

    def run_online(self,
                   case_id: str,
                   llm_diagnosis: str,
                   detector_vote: str,
                   confidence: float = 0.0,
                   reasoning: str = "") -> PipelineResult:
        """
        Online inference for a single case.

        After the LLM produces a diagnosis (via build_uncertainty_prompt),
        the AgreementValidator checks it against the detector vote.
        """
        gate = self.validator.validate(llm_diagnosis, detector_vote, confidence)

        return PipelineResult(
            case_id=case_id,
            llm_diagnosis=llm_diagnosis,
            detector_vote=detector_vote,
            uncertain=(gate.action == "request_review"),
            confidence=confidence,
            reasoning=reasoning,
        )
