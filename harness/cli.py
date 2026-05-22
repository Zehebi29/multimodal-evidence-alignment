"""
Multimodal Evidence Alignment (MEA) — Command-Line Interface

Usage:
    mea build-index  --data ./dataset/  [--detector model.pt]  --output index.json
    mea evaluate     --data ./dataset/  --index index.json     --output results.json
    mea show-config  [--config feature_registry.yaml]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any


def build_index(args):
    """Build multimodal evidence index from a dataset."""
    print(f"Building evidence index...")
    print(f"  Data dir:  {args.data}")
    print(f"  Detector:  {args.detector or 'default (YOLOv5m)'}")
    print(f"  Output:    {args.output}")

    # Load feature registry
    from harness.registry.feature import FeatureRegistry
    config_path = args.config or _default_config()
    registry = FeatureRegistry(config_path)
    print(f"  Registry:  {len(registry.get_all_features())} features loaded")

    # Scan cases
    cases = _find_cases(args.data)
    print(f"  Cases:     {len(cases)} found")
    if not cases:
        print("  ERROR: No cases found. Expected structure: data/CASE_ID/metadata.json")
        sys.exit(1)

    # Build index entry for each case
    index = []
    for i, case_dir in enumerate(cases):
        case_id = os.path.basename(case_dir)
        entry = _build_case_entry(case_id, case_dir, registry, args)
        if entry:
            index.append(entry)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(cases)} cases processed")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"  Done: {len(index)} entries saved to {args.output}")


def evaluate(args):
    """Evaluate using a pre-built evidence index."""
    print(f"Evaluating with evidence index...")
    print(f"  Data dir:  {args.data}")
    print(f"  Index:     {args.index}")
    print(f"  Output:    {args.output}")

    # Load index
    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)
    print(f"  Index:     {len(index)} entries loaded")

    # Load cases
    cases = _find_cases(args.data)
    print(f"  Cases:     {len(cases)} found")

    # Evaluate each case
    results = []
    for i, case_dir in enumerate(cases):
        case_id = os.path.basename(case_dir)
        # Placeholder: in production this runs the full online pipeline
        result = {"case_id": case_id, "status": "pending"}
        results.append(result)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Done: {len(results)} results saved to {args.output}")
    print("  NOTE: Full LLM inference not yet wired — skeleton only.")


def show_config(args):
    """Display the feature registry configuration."""
    from harness.registry.feature import FeatureRegistry
    config_path = args.config or _default_config()
    registry = FeatureRegistry(config_path)
    print(registry.to_summary())


def _default_config() -> str:
    """Find the default feature_registry.yaml."""
    here = Path(__file__).parent
    default = here.parent / "configs" / "feature_registry.yaml"
    if default.exists():
        return str(default)
    # Fallback: search up from CWD
    for parent in Path.cwd().parents:
        candidate = parent / "configs" / "feature_registry.yaml"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Cannot find feature_registry.yaml. "
        "Use --config to specify the path."
    )


def _find_cases(data_dir: str) -> list:
    """Find all case directories under data_dir.
    
    A case directory is identified by containing a metadata.json file.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    cases = []
    for entry in sorted(data_path.iterdir()):
        if entry.is_dir() and (entry / "metadata.json").exists():
            cases.append(str(entry))
        # Also check subdirectories (e.g., data/CASE_ID/metadata.json)
        elif entry.is_dir():
            meta = entry / "metadata.json"
            if meta.exists():
                cases.append(str(entry))
    return cases


def _build_case_entry(case_id: str, case_dir: str, registry, args) -> Dict[str, Any]:
    """Build a single case entry for the evidence index."""
    from harness.registry.extractors import get_extractor
    
    # Extract clinical features
    features = {}
    for name in registry.get_all_features():
        value = registry.extract(name, case_dir)
        if value is not None:
            features[name] = value
    
    # Collect frame predictions (placeholder)
    frames_dir = os.path.join(case_dir, "frames")
    frames = []
    if os.path.isdir(frames_dir):
        frames = sorted(os.listdir(frames_dir))
    
    entry = {
        "case_id": case_id,
        "features": features,
        "num_frames": len(frames),
        "ai_predictions": [],  # populated by detector in full pipeline
        "diagnosis_from_chain": None,
        "evidence_chain": None,
    }
    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Evidence Alignment (MEA) CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mea build-index --data ./my_dataset/ --output index.json
  mea evaluate --data ./my_dataset/ --index index.json --output results.json
  mea show-config --config configs/feature_registry.yaml
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build-index
    p_build = subparsers.add_parser("build-index", help="Build evidence index")
    p_build.add_argument("--data", required=True, help="Dataset directory")
    p_build.add_argument("--detector", default=None, help="Path to YOLO model checkpoint")
    p_build.add_argument("--output", required=True, help="Output index JSON path")
    p_build.add_argument("--config", default=None, help="Feature registry YAML path")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate with evidence index")
    p_eval.add_argument("--data", required=True, help="Dataset directory")
    p_eval.add_argument("--index", required=True, help="Pre-built index JSON")
    p_eval.add_argument("--output", required=True, help="Output results JSON path")
    p_eval.add_argument("--config", default=None, help="Feature registry YAML path")

    # show-config
    p_show = subparsers.add_parser("show-config", help="Display feature registry")
    p_show.add_argument("--config", default=None, help="Feature registry YAML path")

    args = parser.parse_args()

    if args.command == "build-index":
        build_index(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "show-config":
        show_config(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
