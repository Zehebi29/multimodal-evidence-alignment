# Quick Start

## Installation

```bash
git clone https://gitee.com/HangbinZheng/multimodal-evidence-alignment.git
cd multimodal-evidence-alignment
pip install -e .
```

Requirements: Python ≥ 3.9, numpy, scipy, pyyaml.

## LLM Setup

```bash
# 1. Set your API key (never store in config files)
export SILICONFLOW_API_KEY=sk-xxx

# 2. Edit configs/llm_config.yaml to choose provider/model
#    Default: SiliconFlow + DeepSeek V4 Flash
```

## Prepare Your Data

Your dataset must follow this directory structure:

```
my_dataset/
├── CASE-001/
│   ├── metadata.json      # Clinical features
│   └── frames/
│       ├── frame_001.jpg
│       ├── frame_002.jpg
│       └── ...
├── CASE-002/
│   ├── metadata.json
│   └── frames/
│       └── ...
└── ...
```

Each `metadata.json` must contain clinical information. See [Data Format](data-format.md)
for the full schema. Minimal required fields:

```json
{
  "case_id": "CASE-001",
  "clinical_info": {
    "lesion_location": "胃底",
    "echo_pattern": "低",
    "homogeneous": "是"
  }
}
```

## Build an Evidence Index

```bash
mea build-index \
  --data ./my_dataset/ \
  --detector path/to/yolov5m.pt \
  --output ./my_index.json
```

This runs the offline phase:
1. Frame detection (YOLOv5m)
2. Case aggregation (majority vote)
3. Feature extraction (from metadata.json)
4. Evidence index construction

## Evaluate

```bash
mea evaluate \
  --data ./my_dataset/ \
  --index ./my_index.json \
  --output ./results.json
```

The online phase retrieves similar cases from the evidence store and
uses the LLM reasoner to produce case-level evidence chains.

## Programmatic Usage

```python
from harness import MultimodalEvidenceAlignment
from harness.registry.feature import FeatureRegistry

# Load feature registry
registry = FeatureRegistry("configs/feature_registry.yaml")

# Extract features from a case
features = registry.extract_all("./my_dataset/CASE-001/")
print(features)
# → {'lesion_location': '胃底', 'echo_pattern': '低', ...}

# Full analysis
mea = MultimodalEvidenceAlignment()
result = mea.run_online(
    case_id="CASE-001",
    llm_diagnosis="GIST",
    detector_vote="GIST",
    confidence=0.85,
    reasoning="...",
)
```
