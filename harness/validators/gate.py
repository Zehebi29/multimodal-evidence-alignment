"""
LLM-Detector Agreement Validator

Checks whether the LLM's diagnosis agrees with the detector's majority vote.
When these two evidence sources conflict, the case is flagged for expert review.

This is the sole validation rule — no confidence thresholds, no evidence-strength
heuristics. The LLM already integrated all evidence under the Clinical Reasoning
Protocol; the Agreement Validator only catches cases where it reached a conclusion
that contradicts the visual evidence.
"""

import logging
from typing import Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    """Validation result."""
    passed: bool
    action: str                    # "diagnose" | "request_review"
    confidence_adjusted: float
    warnings: List[str] = field(default_factory=list)


class AgreementValidator:
    """Checks LLM-Detector diagnostic agreement.

    When the LLM diagnosis contradicts the detector vote, the case is flagged
    uncertain and routed to expert review. The LLM's reasoning chain is preserved
    intact — it explains *why* the model disagreed with the visual evidence.
    """

    def __init__(self):
        pass

    def validate(self, llm_diagnosis: str, detector_vote: str,
                 confidence: float = 0.0) -> GateDecision:
        """Check agreement between LLM diagnosis and detector majority vote.

        Args:
            llm_diagnosis: LLM's final diagnosis (LM/GIST/NET/EP/LIP/uncertain)
            detector_vote: Detector's majority-vote prediction
            confidence: LLM's self-reported confidence (passed through unchanged)

        Returns:
            GateDecision with action="diagnose" if agreed, "request_review" if conflicted
        """
        # Normalize: "uncertain" means LLM already abstained — pass through
        if llm_diagnosis == "uncertain":
            return GateDecision(
                passed=True,
                action="diagnose",  # LLM already chose to abstain
                confidence_adjusted=confidence,
                warnings=[],
            )

        # Agreement check
        if llm_diagnosis == detector_vote:
            return GateDecision(
                passed=True,
                action="diagnose",
                confidence_adjusted=confidence,
                warnings=[],
            )

        # Conflict: flag for expert review
        logger.info(
            f"LLM-Detector disagreement: LLM={llm_diagnosis}, "
            f"Detector={detector_vote} → flagged for review"
        )
        return GateDecision(
            passed=False,
            action="request_review",
            confidence_adjusted=confidence,
            warnings=[
                f"LLM diagnosis ({llm_diagnosis}) conflicts with "
                f"detector vote ({detector_vote})"
            ],
        )
