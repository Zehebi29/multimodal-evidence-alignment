"""
Cross-frame Consistency Verification for EUS video frames.

Detects inconsistencies across frames in an EUS video sequence using
three complementary signals:
1. Frame-level majority agreement
2. Confidence-weighted vs simple majority vote comparison
3. Frame-level prediction entropy

The goal is to detect cases where frame-level predictions disagree,
indicating that the model is uncertain about the lesion type.
"""
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional


# Thresholds for consistency signals
LOW_AGREEMENT_THRESHOLD = 0.70  # < 70% agreement = low consistency
HIGH_ENTROPY_THRESHOLD = 0.30   # > 30% frames with high entropy = inconsistent
CONFIDENCE_VOTE_DIFF = True     # Whether to check confidence-weighted vs simple vote


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


VALID_CLASSES = {'GIST', 'LM', 'NET', 'EP', 'LIP'}


def compute_frame_entropy(confidence: float, n_classes: int = 5) -> float:
    """Compute approximate frame-level entropy from single prediction confidence.
    
    Uses the confidence as probability of the predicted class, and distributes
    the remaining probability evenly across other classes.
    
    Args:
        confidence: Prediction confidence (0-1)
        n_classes: Number of possible classes
        
    Returns:
        Normalized entropy (0-1)
    """
    remaining = (1.0 - confidence) / max(n_classes - 1, 1)
    probs = [confidence] + [remaining] * (n_classes - 1)
    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
    max_entropy = math.log(n_classes)
    return entropy / max_entropy if max_entropy > 0 else 0


# Signal 1: Majority Agreement
def check_majority_agreement(
    frames: List[Dict],
    threshold: float = LOW_AGREEMENT_THRESHOLD,
) -> Dict:
    """Check if frame-level predictions agree with the majority vote.
    
    Args:
        frames: Frame-level predictions with {class, conf}
        threshold: Minimum agreement threshold
        
    Returns:
        Dict with agreement check results
    """
    if not frames:
        return {'consistent': False, 'agreement': 0.0, 'reason': 'no_frames'}
    
    classes = [_parse_class_name(f.get('class', '?')) for f in frames]
    counter = Counter(classes)
    
    # Filter out invalid classes
    valid_classes = [c for c in classes if c in VALID_CLASSES]
    if not valid_classes:
        return {'consistent': False, 'agreement': 0.0, 'reason': 'no_valid_detections'}
    
    valid_counter = Counter(valid_classes)
    top_class, top_count = valid_counter.most_common(1)[0]
    agreement = top_count / len(valid_classes)
    
    return {
        'consistent': agreement >= threshold,
        'agreement': round(agreement, 3),
        'threshold': threshold,
        'top_class': top_class,
        'top_count': top_count,
        'total_valid': len(valid_classes),
        'signal': 'majority_agreement',
        'reason': None if agreement >= threshold else f'agreement={agreement:.1%} < threshold={threshold:.0%}',
    }


# Signal 2: Confidence-weighted vs Simple Vote
def check_confidence_weighted_vote(
    frames: List[Dict],
) -> Dict:
    """Compare simple majority vote with confidence-weighted vote.
    
    Confidence-weighted vote: sum of confidence for each class, weighted by
    prediction strength. If the top class differs from simple majority, this
    indicates that some frames have very low confidence for their prediction.
    
    Args:
        frames: Frame-level predictions with {class, conf}
        
    Returns:
        Dict with vote comparison results
    """
    if not frames:
        return {'consistent': False, 'simple_vote': '?', 'weighted_vote': '?', 'reason': 'no_frames'}
    
    # Simple majority vote
    classes = [_parse_class_name(f.get('class', '?')) for f in frames]
    valid_pairs = [(c, f.get('conf', 0)) for c, f in zip(classes, frames) if c in VALID_CLASSES]
    
    if not valid_pairs:
        return {'consistent': False, 'reason': 'no_valid_detections'}
    
    # Simple vote
    simple_counter = Counter(c for c, _ in valid_pairs)
    simple_vote = simple_counter.most_common(1)[0][0]
    
    # Confidence-weighted vote
    weighted_scores = {}
    for cls, conf in valid_pairs:
        weighted_scores[cls] = weighted_scores.get(cls, 0) + conf
    weighted_vote = max(weighted_scores, key=weighted_scores.get)
    
    is_consistent = (simple_vote == weighted_vote)
    
    return {
        'consistent': is_consistent,
        'simple_vote': simple_vote,
        'weighted_vote': weighted_vote,
        'simple_counts': dict(simple_counter),
        'weighted_scores': weighted_scores,
        'signal': 'confidence_weighted_vote',
        'reason': None if is_consistent else f'simple={simple_vote} != weighted={weighted_vote}',
    }


# Signal 3: Frame-level Entropy
def check_frame_entropy(
    frames: List[Dict],
    high_entropy_threshold: float = HIGH_ENTROPY_THRESHOLD,
    max_high_entropy_ratio: float = 0.30,
) -> Dict:
    """Check frame-level prediction entropy.
    
    High entropy = the model is uncertain about individual frames.
    If many frames have high entropy, the overall prediction is unreliable.
    
    Args:
        frames: Frame-level predictions with {class, conf}
        high_entropy_threshold: Entropy above this = high entropy
        max_high_entropy_ratio: Max allowed ratio of high-entropy frames
        
    Returns:
        Dict with entropy check results
    """
    if not frames:
        return {'consistent': False, 'mean_entropy': 0.0, 'reason': 'no_frames'}
    
    entropies = []
    for f in frames:
        cls = _parse_class_name(f.get('class', '?'))
        if cls in VALID_CLASSES:
            ent = compute_frame_entropy(f.get('conf', 0.5))
            entropies.append(ent)
    
    if not entropies:
        return {'consistent': False, 'mean_entropy': 0.0, 'reason': 'no_valid_detections'}
    
    mean_entropy = sum(entropies) / len(entropies)
    high_entropy_count = sum(1 for e in entropies if e >= high_entropy_threshold)
    high_entropy_ratio = high_entropy_count / len(entropies)
    
    is_consistent = high_entropy_ratio <= max_high_entropy_ratio
    
    return {
        'consistent': is_consistent,
        'mean_entropy': round(mean_entropy, 3),
        'high_entropy_count': high_entropy_count,
        'high_entropy_ratio': round(high_entropy_ratio, 3),
        'total_valid': len(entropies),
        'signal': 'frame_entropy',
        'reason': None if is_consistent else f'high_entropy_ratio={high_entropy_ratio:.1%} > {max_high_entropy_ratio:.0%}',
    }


# Composite consistency check
def check_consistency(
    frames: List[Dict],
    low_agreement_threshold: float = LOW_AGREEMENT_THRESHOLD,
    high_entropy_threshold: float = HIGH_ENTROPY_THRESHOLD,
    max_high_entropy_ratio: float = 0.30,
) -> Dict:
    """Full consistency verification using all three signals.
    
    Args:
        frames: Raw frame-level predictions
        low_agreement_threshold: Threshold for signal 1
        high_entropy_threshold: Threshold for signal 3
        max_high_entropy_ratio: Max high-entropy frame ratio for signal 3
        
    Returns:
        Dict with overall consistency verdict and per-signal details
    """
    if not frames:
        return {
            'consistent': False,
            'signals': {},
            'triggered_signals': ['no_frames'],
            'severity': 'high',
        }
    
    signals = {}
    triggered = []
    
    # Signal 1: Majority agreement
    sig1 = check_majority_agreement(frames, low_agreement_threshold)
    signals['majority_agreement'] = sig1
    if not sig1['consistent']:
        triggered.append('majority_agreement')
    
    # Signal 2: Confidence-weighted vote
    sig2 = check_confidence_weighted_vote(frames)
    signals['confidence_weighted_vote'] = sig2
    if not sig2['consistent']:
        triggered.append('confidence_weighted_vote')
    
    # Signal 3: Frame entropy
    sig3 = check_frame_entropy(frames, high_entropy_threshold, max_high_entropy_ratio)
    signals['frame_entropy'] = sig3
    if not sig3['consistent']:
        triggered.append('frame_entropy')
    
    # Overall verdict
    n_signals = 3
    n_triggered = len(triggered)
    
    if n_triggered == 0:
        severity = 'none'
        consistent = True
    elif n_triggered == 1:
        severity = 'low'
        consistent = True
    elif n_triggered == 2:
        severity = 'medium'
        consistent = False
    else:
        severity = 'high'
        consistent = False
    
    return {
        'consistent': consistent,
        'signals': signals,
        'triggered_signals': triggered,
        'n_signals_triggered': n_triggered,
        'n_signals_total': n_signals,
        'severity': severity,
    }


def adjust_confidence_for_inconsistency(
    base_confidence: float,
    consistency_result: Dict,
) -> Tuple[float, str]:
    """Adjust confidence score downward based on inconsistency severity.
    
    Args:
        base_confidence: Original confidence score from LLM
        consistency_result: Result from check_consistency()
        
    Returns:
        (adjusted_confidence, adjustment_reason)
    """
    severity = consistency_result.get('severity', 'none')
    triggered = consistency_result.get('triggered_signals', [])
    
    if severity == 'none':
        return base_confidence, 'no_inconsistency'
    
    adjustments = {
        'low': 0.05,
        'medium': 0.15,
        'high': 0.30,
    }
    
    penalty = adjustments.get(severity, 0)
    adjusted = max(base_confidence - penalty, 0.0)
    
    reason = f'consistency_{severity}({"+".join(triggered)})'
    
    return adjusted, reason
