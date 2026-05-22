# Extending MEA

MEA is designed for extensibility. You can add new clinical features,
custom extractors, validators, and evidence weighting rules without
modifying core framework code.

## Choosing a Vector Backend

Evidence Store supports two backends via the `backend` parameter or config:

```python
# Default: in-memory + JSON (zero extra deps, fast for <10K cases)
store = EvidenceStore(api_key="...", backend="json")

# Optional: FAISS IVF index (pip install faiss-cpu, scales to 1M+)
store = EvidenceStore(api_key="...", backend="faiss")
```

Or in `configs/llm_config.yaml`:

```yaml
vector_backend: json   # or faiss
```

| Backend | Deps | Scale | Speed |
|---------|------|-------|-------|
| `json` (default) | None | <10K cases | O(n) brute-force, ~1ms for 500 |
| `faiss` | `faiss-cpu` | 1M+ | IVF + inner product, sub-ms |

## Adding a New Clinical Feature

### 1. Register the feature

Add an entry to `configs/feature_registry.yaml`:

```yaml
features:
  tumor_size:                        # New feature name
    source: doctor                   # auto | visual | doctor | external
    extractor: tumor_size            # Matches extractor function name
    required: false                  # true = must be present
    description: "Tumor dimensions"
    values: []                       # Allowed values (empty = free text)
    default: null
```

### 2. Implement the extractor

Add a function to `harness/registry/extractors/builtin.py`:

```python
def extract_tumor_size(case_dir: str, **kwargs) -> Optional[str]:
    """Extract tumor size from metadata.json."""
    return _from_metadata(case_dir, "tumor_size")
```

### 3. Register the extractor

Add the mapping to `BUILTIN_EXTRACTORS` in the same file:

```python
BUILTIN_EXTRACTORS = {
    # ... existing ...
    "tumor_size": extract_tumor_size,
}
```

### 4. Use it

```python
from harness.registry.feature import FeatureRegistry

registry = FeatureRegistry("configs/feature_registry.yaml")
size = registry.extract("tumor_size", "./my_dataset/CASE-001/")
# → "1.2*1.8"
```

## Adding a Custom Validator

Create a new validator module under `harness/validators/`. For example,
a location checker that validates lesion location against known class
distributions:

```python
# Example: harness/validators/location_check.py
class LocationValidator:
    """Validate that lesion location is consistent with diagnosis."""
    
    VALID_LOCATIONS = {
        "GIST": ["胃底", "胃体"],
        "LM":   ["食管", "胃底", "胃体"],
        "NET":  ["直肠"],
        "EP":   ["胃窦", "胃体", "十二指肠"],
        "LIP":  ["胃窦", "胃体"],
    }
    
    def validate(self, diagnosis: str, location: str) -> bool:
        valid = self.VALID_LOCATIONS.get(diagnosis, [])
        return location in valid
```

## Configuring Evidence Weighting

Feature Mutual Information weights in `feature_registry.yaml` control
how much each feature contributes to the evidence score:

```yaml
lesion_location:
  mi_weight: 1.1497   # Higher = more diagnostically informative

echo_pattern:
  mi_weight: 0.4676   # Lower = less discriminative
```

These weights are derived from training data analysis. You can recalibrate
them for your own dataset by running the feature analysis pipeline.

## Adding a New Clinical Reasoning Rule

The Clinical Reasoning Protocol is encoded in `harness/reasoners/prompts.py`.
To add a new confusion pattern or disease prior, edit the corresponding
section in `PromptTemplates.CLINICAL_CONTEXT`.
