# Data Format

## Dataset Directory Structure

```
dataset/
└── CASE_ID/              # One directory per case
    ├── metadata.json     # Clinical features (required)
    └── frames/           # EUS frame images (optional)
        ├── frame_01.jpg
        ├── frame_02.jpg
        └── ...
```

## metadata.json Schema

The `metadata.json` file provides clinical context for each case.
Fields can be at the top level or nested under `clinical_info`.

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `case_id` | string | Unique case identifier | `"CASE-V1_1_P50"` |
| `lesion_location` | string | Lesion anatomical location | `"胃底"`, `"胃体"`, `"食管"` |
| `echo_pattern` | string | Ultrasound echogenicity | `"低"` (hypoechoic), `"高"` (hyperechoic) |
| `homogeneous` | string | Echo homogeneity | `"是"` (homogeneous), `"否"` (heterogeneous) |

### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `layer_origin` | string | Lesion layer of origin | `"固有肌层"`, `"粘膜下层"` |
| `tumor_size` | string | Tumor dimensions | `"1.2*1.8"` |
| `border` | string | Lesion border character | `"光滑"`, `"不规则"` |
| `shape` | string | Lesion shape | `"规则"`, `"不规则"` |
| `age` | string/number | Patient age | `"52"` |
| `gender` | string | Patient gender | `"男"`, `"女"` |
| `pathology` | string | Gold-standard pathology | `"GIST"`, `"LM"` |

### Full Example

```json
{
  "case_id": "CASE-V1_1_P50",
  "source": "EUS",
  "source_case_id": "smt1_p50",
  "pathology": "EP",
  "frame_count": 6,
  "clinical_info": {
    "age": "nan",
    "gender": "nan",
    "lesion_location": "胃体",
    "tumor_size": "1.2*1.8",
    "layer_origin": "固有肌层",
    "layer_confidence": "高",
    "echo_pattern": "低",
    "homogeneous": "否",
    "border": "光滑",
    "shape": "规则"
  }
}
```

## Evidence Index Format

The evidence index (`index.json`) is a JSON array of case entries:

```json
[
  {
    "case_id": "CASE-V1_1_P0",
    "features": {
      "lesion_location": "食管",
      "echo_pattern": "低",
      "homogeneous": "是",
      "layer_origin": "粘膜肌层"
    },
    "num_frames": 4,
    "ai_predictions": [...],
    "diagnosis_from_chain": "LM",
    "evidence_chain": {
      "reasoning": "...",
      "supporting_evidence": [...],
      "opposing_evidence": [...],
      "differential": [...]
    }
  }
]
```

The full evidence chain follows an 8-field typed schema — see
`harness/slots/audit_trail.py` for the `EvidenceChain` dataclass.
