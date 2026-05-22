"""
Key Frame Selection for EUS video frames.

Selects the most diagnostically useful frames from an EUS video sequence by:
1. Quality filtering (removing low-confidence frames)
2. Diversity selection (covering different prediction patterns)

EUS videos are not continuous temporal sequences - the probe is moved and rotated
to capture different angles of the same lesion. Each frame may show the lesion
from a different viewpoint with different quality.
"""
import math
from collections import Counter
from typing import List, Dict, Any, Optional

# Minimum confidence threshold for quality filter
MIN_CONFIDENCE = 0.3
# Maximum number of frames to select
MAX_KEY_FRAMES = 5
# Default if we want to keep original behavior (use all frames)
USE_ALL_FRAMES = 0


def quality_filter(
    frames_with_preds: List[Dict],
    min_conf: float = MIN_CONFIDENCE,
) -> List[Dict]:
    """Filter out low-quality frames (low detection confidence).
    
    Args:
        frames_with_preds: List of {frame_id, class, conf} dicts
        min_conf: Minimum confidence threshold
        
    Returns:
        Filtered list of frames
    """
    if not frames_with_preds:
        return []
    
    # Skip frames with no detection or very low confidence
    filtered = [f for f in frames_with_preds if f.get('conf', 0) >= min_conf]
    
    return filtered


def _parse_class_name(cls: str) -> str:
    """Parse YOLO class name/number to standardized form."""
    CLASS_MAP = {
        '0': 'GIST', '1': 'LM', '2': '?', '3': 'NET', '4': 'EP', '5': 'LIP',
        '6': 'Other', '7': 'Other',
        'gist': 'GIST', 'lm': 'LM', 'net': 'NET', 'ep': 'EP', 'lip': 'LIP',
    }
    if cls in CLASS_MAP:
        return CLASS_MAP[cls]
    if cls in ('GIST', 'LM', 'NET', 'EP', 'LIP'):
        return cls
    return '?'


def diversity_selection(
    frames: List[Dict],
    max_frames: int = MAX_KEY_FRAMES,
) -> List[Dict]:
    """Select diverse frames by clustering on prediction patterns.
    
    Groups frames by their predicted class, then selects frames from each
    group proportionally. Within each group, prefers higher-confidence frames.
    
    Args:
        frames: Filtered frame list
        max_frames: Maximum number of frames to select (0 = keep all)
        
    Returns:
        Selected diverse frames
    """
    if not frames:
        return []
    
    if max_frames <= 0 or max_frames >= len(frames):
        return frames
    
    # Group frames by predicted class
    groups = {}
    for f in frames:
        cls = _parse_class_name(f.get('class', '?'))
        if cls not in groups:
            groups[cls] = []
        groups[cls].append(f)
    
    # Sort each group by confidence (descending)
    for cls in groups:
        groups[cls].sort(key=lambda x: -x.get('conf', 0))
    
    # Allocate slots to each group proportionally
    n_groups = len(groups)
    if n_groups == 1:
        return groups[list(groups.keys())[0]][:max_frames]
    
    # Proportional allocation: at least 1 per group, rest by size
    allocations = {}
    for cls in groups:
        allocations[cls] = 1  # each group gets at least 1
    
    remaining = max_frames - len(groups)
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    
    for cls, group in sorted_groups:
        if remaining <= 0:
            break
        extra = min(len(group) - 1, max(1, int(remaining / max(1, n_groups))))
        allocations[cls] += extra
        remaining -= extra
    
    # Select frames
    selected = []
    for cls in sorted(allocations.keys()):
        n = min(allocations[cls], len(groups[cls]))
        selected.extend(groups[cls][:n])
    
    # Sort by original order (by frame_id if available)
    selected.sort(key=lambda x: x.get('frame_id', ''))
    
    return selected


def select_key_frames(
    frames: List[Dict],
    min_conf: float = MIN_CONFIDENCE,
    max_frames: int = MAX_KEY_FRAMES,
) -> List[Dict]:
    """Complete key frame selection pipeline.
    
    Args:
        frames: Raw frame-level predictions
        min_conf: Minimum confidence for quality filter
        max_frames: Max frames after selection (0 = all filtered frames)
        
    Returns:
        Selected key frames
    """
    # Step 1: Quality filter
    filtered = quality_filter(frames, min_conf)
    
    if not filtered:
        return []
    
    # Step 2: Diversity selection
    selected = diversity_selection(filtered, max_frames)
    
    return selected


def get_keyframe_vote(
    frames: List[Dict],
    min_conf: float = MIN_CONFIDENCE,
    max_frames: int = MAX_KEY_FRAMES,
) -> Dict:
    """Get majority vote computed from key frames only.
    
    Returns:
        Dictionary with vote details
    """
    key_frames = select_key_frames(frames, min_conf, max_frames)
    
    if not key_frames:
        return {
            'vote': '?',
            'agreement': 0.0,
            'mean_confidence': 0.0,
            'n_key_frames': 0,
            'n_original_frames': len(frames),
            'n_filtered_out': len(frames),
        }
    
    classes = [_parse_class_name(f.get('class', '?')) for f in key_frames]
    confs = [f.get('conf', 0) for f in key_frames]
    
    counter = Counter(classes)
    top_class = counter.most_common(1)[0][0] if counter else '?'
    agreement = counter.most_common(1)[0][1] / len(key_frames) if key_frames else 0
    mean_conf = sum(confs) / len(confs) if confs else 0
    
    return {
        'vote': top_class,
        'agreement': agreement,
        'mean_confidence': mean_conf,
        'n_key_frames': len(key_frames),
        'n_original_frames': len(frames),
        'n_filtered_out': len(frames) - len(key_frames),
        'class_distribution': dict(counter),
    }
