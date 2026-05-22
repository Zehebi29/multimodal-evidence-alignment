"""
Prompt Templates - Structured prompt system for Multimodal Evidence Alignment (MEA).

Three-layer LLM control:
- This file: Control INPUT (prompt templates)
- client.py: Control EXECUTION (API calls)
- parsers.py + gate.py: Control OUTPUT (validation + gating)
"""

from typing import Dict, List, Any


class PromptTemplates:
    """Prompt template collection."""

    CLINICAL_CONTEXT = """You are a Clinical Evidence Analysis System for EUS (Endoscopic Ultrasound).
Your task is to integrate AI frame-level predictions with clinical features to produce a case-level diagnosis.

Disease Categories (based on real clinical data distributions):
- LM (Leiomyoma): Most common in esophagus (55%) or gastric fundus/body (45%). Layer: muscularis propria (46%) OR muscularis mucosae (44%). Echo: hypoechoic. Homogeneous: 93%.
- GIST (GI Stromal Tumor): Gastric fundus (67%) or body (25%). Layer: muscularis propria (91%). Echo: hypoechoic. Heterogeneous (50%) or homogeneous (50%).
- NET (Neuroendocrine Tumor): Rectum (94%). Layer: muscularis mucosae (48%) or submucosa (45%). Echo: hypoechoic. Homogeneous: 76%.
- EP (Ectopic Pancreas): Gastric antrum (69%) or body (23%). Layer: submucosa (54%), muscularis propria (14%), or muscularis mucosae (11%). Echo: mostly hypoechoic (74%), can be mixed/hyperechoic. Heterogeneous: 83%.
- LIP (Lipoma): Gastric antrum (61%) or body (22%). Layer: submucosa (95%). Echo: hyperechoic (95%). Homogeneous: 95%.

Common Confusion Patterns (from real misdiagnoses):
- LM→GIST confusion (6 cases): When AI predicts LM but some frames suggest GIST, check location carefully. LM is esophagus/stomach (93% homogeneous); GIST is gastric fundus (67%) with muscularis propria origin. If location is NOT rectum and homogeneity is high, prefer LM.
  Example: location=胃底, AI predictions=[LM:1], AI said LM but correct is GIST
  Example: location=胃体, AI predictions=[LM:2], AI said LM but correct is GIST
- uncertain→LM confusion (3 cases): When AI frames are split but clinical features suggest LM (esophagus/stomach location, homogeneous hypoechoic), commit to LM even with moderate frame agreement.
  Example: location=食管, AI predictions=[LM:1, NET:1], AI said uncertain but correct is LM
  Example: location=胃体, AI predictions=[GIST:1, LM:1], AI said uncertain but correct is LM
- uncertain→LIP confusion (3 cases): Review location and echo pattern carefully.
  Example: location=升结肠, AI predictions=[LIP:1], AI said uncertain but correct is LIP
  Example: location=乙状结肠, AI predictions=[LIP:1], AI said uncertain but correct is LIP
- GIST→LM confusion (3 cases): When AI predicts GIST but correct is LM, the key is layer origin: GIST=91% muscularis propria, LM=46% muscularis propria + 44% muscularis mucosae. If the lesion appears to be in mucosal/submucosal layer, prefer LM.
  Example: location=胃底, AI predictions=[GIST:1], AI said GIST but correct is LM
  Example: location=胃底, AI predictions=[GIST:2, LIP:1], AI said GIST but correct is LM
- LM→EP confusion (2 cases): EP is in gastric antrum (69%), heterogeneous (83%), can be mixed/hyperechoic. If location is antrum and texture is heterogeneous, consider EP.
  Example: location=胃窦, AI predictions=[LM:1], AI said LM but correct is EP
  Example: location=胃窦, AI predictions=[LM:1], AI said LM but correct is EP
- NET→GIST confusion (2 cases): NET is typically rectum (94%), GIST is gastric fundus (67%). Location is the strongest discriminator.
  Example: location=直肠, AI predictions=[GIST:1], AI said NET but correct is GIST
  Example: location=胃体, AI predictions=[NET:1], AI said NET but correct is GIST
- uncertain→NET confusion (2 cases): Review location and echo pattern carefully.
  Example: location=直肠, AI predictions=[LM:4], AI said uncertain but correct is NET
  Example: location=直肠, AI predictions=[LIP:2], AI said uncertain but correct is NET
- uncertain→EP confusion (1 cases): Review location and echo pattern carefully.
  Example: location=十二指肠, AI predictions=[GIST:1, LIP:1], AI said uncertain but correct is EP
- NET→LM confusion (1 cases): Review location and echo pattern carefully.
  Example: location=横结肠, AI predictions=[NET:3], AI said NET but correct is LM
- GIST→EP confusion (1 cases): Review location and echo pattern carefully.
  Example: location=胃角, AI predictions=[GIST:1], AI said GIST but correct is EP
- uncertain→GIST confusion (1 cases): Review location and echo pattern carefully.
  Example: location=直肠, AI predictions=[GIST:1], AI said uncertain but correct is GIST
- NET→EP confusion (1 cases): Review location and echo pattern carefully.
  Example: location=十二指肠, AI predictions=[NET:2], AI said NET but correct is EP
- EP→LIP confusion (1 cases): Review location and echo pattern carefully.
  Example: location=回盲部, AI predictions=[EP:1], AI said EP but correct is LIP

CRITICAL INSTRUCTIONS:
1. AI frame-level predictions are the PRIMARY diagnostic signal. When 3+ frames agree with >70% confidence, this is strong evidence.
2. Clinical features are SUPPLEMENTARY. They help refine the diagnosis and build the evidence chain, but should NOT override consistent AI predictions.
3. Clinical feature measurements may have imprecision — slight deviations from textbook descriptions are NORMAL and EXPECTED. Do NOT treat minor feature mismatches as strong counter-evidence.
4. Only choose "uncertain" when there is genuine conflicting evidence: AI predictions themselves are split (no clear majority), AND clinical features strongly contradict the leading diagnosis.
5. When AI predictions are highly consistent, you SHOULD commit to a diagnosis. This is your job — to integrate evidence and make a judgment.
6. You ARE performing case-level diagnosis based on evidence. This is evidence-based clinical reasoning, not guesswork."""

    @classmethod
    def build_evidence_chain_prompt(cls, case_id, predictions, features, ground_truth=None):
        """Build evidence chain construction prompt (bilingual output)."""
        pred_lines = []
        for p in predictions:
            pred_lines.append("  - {}: {} (confidence {:.1%})".format(
                p['frame_id'], p['class'], p['conf']))
        pred_text = "\n".join(pred_lines)

        feat_lines = []
        cal_context = None
        for k, v in features.items():
            if k == "_calibration_context":
                cal_context = v
            else:
                feat_lines.append("  - {}: {}".format(k, v))
        feat_text = "\n".join(feat_lines) if feat_lines else "  (no clinical features)"

        from collections import Counter
        votes = Counter(p["class"] for p in predictions)
        vote_text = ", ".join("{}: {} votes".format(cls, cnt) for cls, cnt in votes.most_common())

        # 计算 AI 一致性
        total = len(predictions)
        top_cls, top_cnt = votes.most_common(1)[0]
        agreement = top_cnt / total if total > 0 else 0
        agreement_pct = f"{agreement:.0%}"

        system = cls.CLINICAL_CONTEXT

        user = """## Task: Build clinical evidence chain for case {}

### Input Data

**Ground Truth Diagnosis**: {ground_truth}

**Frame-level AI Predictions** ({} frames total):
{}

**Vote Summary**: {} (top class agreement: {})

**Clinical Features**:
{}
{cal_section}
### Instructions

You are given the ground truth diagnosis above. Your task is to synthesize the AI predictions and clinical features into a structured evidence chain that justifies this diagnosis. Identify which findings support it and which findings challenge it (every case has both). List plausible alternative diagnoses with probability estimates.

### Output Requirements

Output strict JSON with exactly five fields:

```json
{{
  "diagnosis": "{ground_truth_json}",
  "reasoning": "clinical reasoning in Chinese (2-3 sentences, integrate AI predictions and clinical features)",
  "supporting_evidence": [
    {{"type": "frame|feature|consistency", "description": "evidence description in Chinese", "strength": "strong|moderate|weak"}}
  ],
  "opposing_evidence": [
    {{"type": "frame|feature|conflict", "description": "counter-evidence description in Chinese", "severity": "high|medium|low"}}
  ],
  "differential": [
    {{"diagnosis": "differential diagnosis", "probability": 0.0-1.0, "reason": "reasoning in Chinese"}}
  ]
}}
```

IMPORTANT:
- The "diagnosis" field MUST be exactly "{ground_truth_json}" — this is the known correct answer
- All text fields (reasoning, supporting_evidence, opposing_evidence, differential) MUST be in Chinese
- Every case should have at least one supporting and one opposing evidence item
- The differential list must contain at least 2 alternative diagnoses with realistic probabilities summing to ≤1.0""".format(
            case_id, ground_truth, len(predictions), pred_text, vote_text, agreement_pct, feat_text,
            ground_truth=ground_truth, ground_truth_json=ground_truth,
            cal_section="\n\n**Calibration Context** (from historical cases):\n" + cal_context if cal_context else "")

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ── Uncertainty Detection Prompt (for error detection experiment) ──
    
    UNCERTAINTY_SYSTEM = """You are an AI evaluation system for EUS (Endoscopic Ultrasound) diagnosis.

Your task is to evaluate AI frame-level predictions and clinical features, then output your best diagnosis and confidence level.

Output categories: LM, GIST, NET, EP, LIP, uncertain"""

    @classmethod
    def build_uncertainty_prompt(cls, case_id, predictions, features, cal_context=None):
        """Build uncertainty detection prompt (baseline: no RAG context)."""
        pred_lines = []
        for p in predictions:
            pred_lines.append("  - {}: {} (confidence {:.1%})".format(
                p['frame_id'], p['class'], p['conf']))
        pred_text = "\n".join(pred_lines)

        feat_lines = []
        for k, v in features.items():
            if k == "_calibration_context":
                continue
            feat_lines.append("  - {}: {}".format(k, v))
        feat_text = "\n".join(feat_lines) if feat_lines else "  (no clinical features)"

        from collections import Counter
        votes = Counter(p["class"] for p in predictions)
        vote_text = ", ".join("{}: {} votes".format(c, cnt) for c, cnt in votes.most_common())

        total = len(predictions)
        top_cls, top_cnt = votes.most_common(1)[0]
        agreement = top_cnt / total if total > 0 else 0
        agreement_pct = "{:.0%}".format(agreement)

        system = cls.UNCERTAINTY_SYSTEM

        user = """## Task: Evaluate uncertainty for case {case_id}

### AI Frame-level Predictions ({n_frames} frames)
{pred_text}

**Vote Summary**: {vote_text} (top class agreement: {agreement})

### Clinical Features
{feat_text}
{cal_section}

### Instructions

1. Examine the AI predictions:
   - Are they consistent? (high agreement = confident, low agreement = uncertain)
   - Do frame confidences vary widely?

2. Examine clinical features:
   - Do they align with the AI's top prediction?
   - Are there contradictions?

3. {rag_instruction}

4. Output your assessment:

```json
{{
  "diagnosis": "LM/GIST/NET/EP/LIP/uncertain",
  "confidence": 0.0-1.0,
  "needs_review": true/false,
  "reasoning": "brief explanation (1-2 sentences)"
}}
```

IMPORTANT:
- Choose "uncertain" when evidence is genuinely conflicting or insufficient
- Set "needs_review": true when diagnosis is "uncertain" OR confidence < 0.7
- ALL text in Chinese (except JSON keys)""".format(
            case_id=case_id,
            n_frames=len(predictions),
            pred_text=pred_text,
            vote_text=vote_text,
            agreement=agreement_pct,
            feat_text=feat_text,
            cal_section="\n\n" + cal_context if cal_context else "",
            rag_instruction="You have no external reference cases. Rely on AI predictions and clinical features." if not cal_context else "Consider the reference cases below for additional context.",
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        """Build pattern discovery prompt."""
        fb_lines = []
        for fb in feedback_data[:20]:
            ai = fb.get("ai_diagnosis", "?")
            correct = fb.get("correct_diagnosis", "?")
            error_type = fb.get("error_type", "?")
            features = fb.get("features", {})
            feat_str = ", ".join("{}={}".format(k, v) for k, v in features.items()) if features else "none"
            fb_lines.append("  - AI: {} -> Correct: {} | Error: {} | Features: {}".format(
                ai, correct, error_type, feat_str))
        fb_text = "\n".join(fb_lines)

        rules_text = ""
        if current_rules:
            import json
            rules_text = "\n### Current Rules\n```json\n{}\n```".format(
                json.dumps(current_rules, ensure_ascii=False, indent=2))

        system = cls.CLINICAL_CONTEXT

        user = """## Task: Analyze feedback data and discover error patterns

### Feedback Data ({} total, showing first {})
{}{}

### Requirements

Analyze error patterns in the feedback. Output strict JSON:

```json
{{
  "patterns": [
    {{
      "pattern_id": "unique_id",
      "type": "confusion_pair|feature_misuse|threshold_issue|systematic_bias",
      "description": "pattern description",
      "evidence": "data evidence supporting this pattern",
      "frequency": occurrence_count,
      "severity": "high|medium|low",
      "root_cause": "root cause analysis",
      "affected_diagnoses": ["affected diagnosis categories"]
    }}
  ],
  "summary": "overall analysis summary (3-5 sentences)",
  "priority_ranking": ["pattern_ids ranked by importance"]
}}
```

IMPORTANT:
1. Focus on high-frequency errors (confusion pairs >= 2 occurrences)
2. Analyze systemic causes, not just statistics
3. Consider correlation between features and errors
4. Distinguish model issues from feature fusion issues
5. ALL text must be in Chinese""".format(
            len(feedback_data), min(len(feedback_data), 20),
            fb_text, rules_text, "")

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @classmethod
    def build_evolution_prompt(cls, patterns, feedback_stats, current_config):
        """Build evolution suggestion prompt."""
        import json
        patterns_text = json.dumps(patterns, ensure_ascii=False, indent=2)
        config_text = json.dumps(current_config, ensure_ascii=False, indent=2)
        stats_text = json.dumps(feedback_stats, ensure_ascii=False, indent=2)

        system = cls.CLINICAL_CONTEXT

        user = """## Task: Propose strategy evolution based on discovered patterns

### Discovered Patterns
```json
{}
```

### Feedback Statistics
```json
{}
```

### Current Configuration
```json
{}
```

### Requirements

Propose specific strategy evolution suggestions. Output strict JSON:

```json
{{
  "suggestions": [
    {{
      "suggestion_id": "unique_id",
      "title": "suggestion title",
      "description": "detailed description",
      "type": "weight_adjustment|threshold_change|feature_add|rule_change|prompt_update",
      "target": "config path to modify (e.g., feature_importance.echo_pattern)",
      "current_value": "current value",
      "proposed_value": "proposed new value",
      "reasoning": "why this change (based on which data and patterns)",
      "expected_impact": "expected effect",
      "risk_level": "low|medium|high",
      "priority": 1-5
    }}
  ],
  "reasoning": "overall evolution strategy (2-3 sentences)",
  "config_patches": {{
    "config_path": "new_value"
  }}
}}
```

IMPORTANT:
1. Suggestions must be specific and actionable (give exact values)
2. High-risk changes need rollback plans
3. config_patches are direct patches to config file
4. Prioritize high-frequency, high-severity issues
5. ALL text must be in Chinese""".format(patterns_text, stats_text, config_text)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @classmethod
    def build_case_review_prompt(cls, case_id, original_diagnosis, doctor_correction, correction_details=None):
        """Build case review analysis prompt."""
        details_text = ""
        if correction_details:
            import json
            details_text = "\n### Correction Details\n```json\n{}\n```".format(
                json.dumps(correction_details, ensure_ascii=False, indent=2))

        system = cls.CLINICAL_CONTEXT

        user = """## Task: Analyze doctor's review correction for case {}

### AI Original Diagnosis
{}

### Doctor's Correction
{}

### Requirements

Analyze the significance of this correction. Output JSON:

```json
{{
  "error_type": "false_positive|false_negative|wrong_subtype|feature_error",
  "severity": "high|medium|low",
  "root_cause": "root cause of the error",
  "learning": "what can be learned from this correction",
  "affected_features": ["which feature judgments were wrong"],
  "suggested_action": "suggested improvement action"
}}
```""".format(case_id, original_diagnosis, doctor_correction, details_text)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
