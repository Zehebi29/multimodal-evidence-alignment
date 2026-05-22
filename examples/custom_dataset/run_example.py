"""
Multimodal Evidence Alignment (MEA) — Minimal Example

This script demonstrates the core MEA workflow using the smt_1-2 dataset:
1. Load feature registry
2. Extract clinical features from a case
3. Run majority vote aggregation
4. Show evidence chain schema

Usage:
    cd multimodal-evidence-alignment
    python examples/custom_dataset/run_example.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.registry.feature import FeatureRegistry
from harness.tools.aggregator import aggregate_majority_vote


def main():
    # ── Configuration ──
    config_path = str(Path(__file__).parent.parent.parent / "configs" / "feature_registry.yaml")
    data_dir = "/data/eus/datasets/smt_1-2/cases"
    
    print("=" * 60)
    print("Multimodal Evidence Alignment (MEA) — Example")
    print("=" * 60)
    
    # ── 1. Load Feature Registry ──
    print("\n[1] Feature Registry")
    registry = FeatureRegistry(config_path)
    print(f"    Loaded {len(registry.get_all_features())} features:")
    for name, feat in registry.get_all_features().items():
        print(f"      - {name}: {feat.description} (source={feat.source}, mi_weight={feat.mi_weight})")
    
    # ── 2. Find sample cases ──
    print("\n[2] Dataset")
    cases = [d for d in sorted(os.listdir(data_dir))
             if os.path.isdir(os.path.join(data_dir, d))]
    print(f"    Found {len(cases)} cases")
    
    # Pick 3 diverse cases for demonstration
    demo_cases = []
    for case_id in cases:
        case_dir = os.path.join(data_dir, case_id)
        features = registry.extract_all(case_dir)
        if len(demo_cases) >= 3:
            break
        if features:
            demo_cases.append((case_id, case_dir, features))
    
    # ── 3. Feature Extraction ──
    print("\n[3] Feature Extraction (3 sample cases)")
    for case_id, case_dir, features in demo_cases:
        print(f"    {case_id}:")
        for k, v in features.items():
            print(f"      {k}: {v}")
    
    # ── 4. Case Aggregation ──
    print("\n[4] Case Aggregation (Majority Vote)")
    results_dir = "/data/eus/results/smt_1-2"
    for case_id, case_dir, _ in demo_cases:
        det_file = os.path.join(results_dir, f"{case_id}.json")
        if not os.path.exists(det_file):
            print(f"    {case_id}: no results file, skipping")
            continue
        
        with open(det_file) as f:
            case_result = json.load(f)
        
        result = aggregate_majority_vote(case_result)
        print(f"    {case_id}:")
        print(f"      dominant_class: {result['dominant_class']}")
        print(f"      confidence:     {result['confidence']:.1%}")
        print(f"      consistency:    {result['consistency']:.1%}")
        print(f"      votes:          {dict(result['class_votes'])}")
    
    # ── 5. Evidence Chain Schema ──
    print("\n[5] Evidence Chain Schema (8-field typed)")
    from harness.slots.audit_trail import EvidenceChain
    import inspect
    for field_name, field_obj in EvidenceChain.__dataclass_fields__.items():
        ftype = field_obj.type
        print(f"    {field_name}: {ftype}")
    
    # ── 6. CLI equivalent ──
    print("\n[6] CLI Commands")
    print("    # Build index:")
    print(f"    mea build-index --data {data_dir} --output /tmp/mea_demo_index.json")
    print("    # Show config:")
    print("    mea show-config")
    
    print("\n" + "=" * 60)
    print("Example complete. See docs/ for full documentation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
