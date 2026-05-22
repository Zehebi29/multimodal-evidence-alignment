"""
Doctor Interaction — 医生交互机制 (stub)

Note: The paper's experimental pipeline does not use an interactive clinical
interaction system. Online inference is: LLM → AgreementValidator → Gate/Report.

This module is a placeholder for future interactive clinical deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class InteractionResponse:
    """Stub interaction response."""
    interaction_type: str = "diagnose"
    diagnosis: Optional[str] = None
    confidence: Optional[float] = None
    candidates: List[Dict] = field(default_factory=list)
    suggestions: List[Dict] = field(default_factory=list)
    required_info: List[str] = field(default_factory=list)
    report: Optional[Dict] = None


@dataclass
class ReviewForm:
    """Stub review form."""
    case_id: str
    feedback_id: str
    ai_diagnosis: str
    ai_confidence: float
    ai_candidates: List[Dict] = field(default_factory=list)
    ai_sufficiency: float = 0.0
    ai_reasoning: Dict = field(default_factory=dict)
    fields: List[Dict] = field(default_factory=list)


class ClinicalInteraction:
    """Stub — not used in the paper's experimental pipeline."""
    def __init__(self, *args, **kwargs):
        pass
