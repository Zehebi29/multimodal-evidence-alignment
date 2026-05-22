# Multimodal Evidence Alignment (MEA) — User Guide

Welcome to the Multimodal Evidence Alignment (MEA) documentation.

MEA identifies cases where frame-level AI predictions are likely erroneous
through multimodal evidence retrieval, protocol-constrained LLM reasoning,
and cross-model agreement validation. It produces structured, auditable
diagnostic reports with explicit evidence chains and differential diagnoses.

## Quick Links

- [Quick Start](quickstart.md) — Install, prepare data, run your first pipeline
- [Data Format](data-format.md) — Dataset directory structure and metadata schema
- [Extending MEA](extending.md) — Add new features, extractors, or validators
- [Architecture](architecture.md) — Component DAG and data flow

## Command Line Interface

```bash
# Build an evidence index from your dataset
mea build-index --data ./my_dataset/ --output index.json

# Evaluate using a pre-built index
mea evaluate --data ./my_dataset/ --index index.json --output results.json

# Inspect the feature registry
mea show-config
```

## Programmatic API

```python
from harness import MultimodalEvidenceAlignment

mea = MultimodalEvidenceAlignment()

result = mea.run_online(
    case_id="CASE-001",
    llm_diagnosis="GIST",
    detector_vote="GIST",
    confidence=0.85,
    reasoning="...",
)
print(result.uncertain)  # False = diagnosis issued
```
