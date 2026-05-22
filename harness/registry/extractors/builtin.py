"""
Built-in feature extractors for Multimodal Evidence Alignment (MEA).

Each extractor is a callable that takes a case context and returns a feature value.
New extractors can be added by:
1. Creating a new module in this directory
2. Registering it in feature_registry.yaml with `extractor: module.function`
"""

from typing import Any, Dict, Optional
import json
import os


def extract_lesion_location(case_dir: str, **kwargs) -> Optional[str]:
    """Extract lesion location from metadata.json (auto-provided by EUS device)."""
    return _from_metadata(case_dir, "lesion_location")


def extract_echo_pattern(case_dir: str, **kwargs) -> Optional[str]:
    """Extract echo pattern from metadata.json (doctor-annotated)."""
    return _from_metadata(case_dir, "echo_pattern")


def extract_homogeneous(case_dir: str, **kwargs) -> Optional[str]:
    """Extract homogeneity from metadata.json (doctor-annotated)."""
    return _from_metadata(case_dir, "homogeneous")


def extract_layer_origin(case_dir: str, **kwargs) -> Optional[str]:
    """Extract layer origin from metadata.json (doctor-annotated)."""
    return _from_metadata(case_dir, "layer_origin")


def extract_gender(case_dir: str, **kwargs) -> Optional[str]:
    """Extract patient gender from metadata.json."""
    return _from_metadata(case_dir, "gender")


def extract_age(case_dir: str, **kwargs) -> Optional[str]:
    """Extract patient age from metadata.json."""
    return _from_metadata(case_dir, "age")


def _from_metadata(case_dir: str, key: str) -> Optional[str]:
    """Read a field from the case's metadata.json.
    
    Handles both flat metadata and nested clinical_info structures.
    """
    meta_path = os.path.join(case_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Try flat key first
        if key in meta:
            return meta[key]
        # Try nested under clinical_info
        ci = meta.get("clinical_info", {})
        if key in ci:
            return ci[key]
        return None
    except (json.JSONDecodeError, IOError):
        return None


# Registry of all built-in extractors
BUILTIN_EXTRACTORS = {
    "lesion_location": extract_lesion_location,
    "echo_pattern": extract_echo_pattern,
    "homogeneous": extract_homogeneous,
    "layer_origin": extract_layer_origin,
    "gender": extract_gender,
    "age": extract_age,
}


def get_extractor(name: str):
    """Look up a built-in extractor by feature name."""
    return BUILTIN_EXTRACTORS.get(name)
