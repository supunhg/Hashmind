"""Main identification interface."""

import functools
from typing import Dict, Any, List, Optional
from .core.detector import DetectorPipeline
from .features.extractor import FeatureExtractor
from .ml.classifier import MLClassifier
from .ml.confidence import ConfidenceFuser


class IdentificationResult:
    """Structured result from identification."""
    
    def __init__(self, matches: List[Dict[str, Any]], metadata: Dict[str, Any], ml_used: bool = False):
        """
        Initialize result.
        
        Args:
            matches: List of algorithm matches with confidence scores
            metadata: Additional analysis metadata
            ml_used: Whether ML classification was used
        """
        self.matches = matches
        self.metadata = metadata
        self.ml_used = ml_used
    
    def top_match(self) -> Optional[str]:
        """Get the most likely algorithm."""
        if self.matches:
            return self.matches[0]['algorithm']
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'matches': self.matches,
            'metadata': self.metadata,
            'top_match': self.top_match(),
            'ml_used': self.ml_used
        }
    
    def __str__(self) -> str:
        """String representation."""
        if not self.matches:
            return "No matches found"
        
        lines = ["Identification Results:"]
        if self.ml_used:
            lines.append("(ML-Enhanced Detection)")
        
        for i, match in enumerate(self.matches[:5], 1):  # Top 5
            lines.append(f"{i}. {match['algorithm']} (confidence: {match['confidence']:.2%})")
            lines.append(f"   {match['reason']}")
        
        return "\n".join(lines)


# Global instances (lazy loaded)
_ml_classifier = None
_detector = None
_feature_extractor = None

# Cache for results (prevents recomputation)
@functools.lru_cache(maxsize=1000)
def _cached_identify(input_string: str, use_ml_flag: bool) -> tuple:
    """Cached version of identification (returns tuple for hashability)."""
    result = _identify_internal(input_string, use_ml=use_ml_flag)
    # Convert to tuple for caching
    return (tuple(tuple(m.items()) for m in result.matches), 
            tuple(result.metadata.items()), 
            result.ml_used)


def _identify_internal(input_string: str, 
                       context: Optional[Dict[str, Any]] = None,
                       use_ml: bool = True) -> IdentificationResult:
    """
    Identify hash or format type.
    
    This is the main entry point for the hashmind library.
    Combines fast heuristic detection with optional ML classification.
    
    Args:
        input_string: String to identify
        context: Optional context for better identification
        use_ml: Whether to use ML classification (default: True, falls back if unavailable)
        
    Returns:
        IdentificationResult with matches and metadata
        
    Example:
        >>> from hashmind import identify
        >>> result = identify("5d41402abc4b2a76b9719d911017c592")
        >>> print(result.top_match())
        md5_hex
    """
    global _ml_classifier, _detector, _feature_extractor
    
    # Phase 1: Heuristic detection (always runs)
    # Use cached detector instance
    if _detector is None:
        _detector = DetectorPipeline()
    heuristic_result = _detector.analyze(input_string)
    
    # Convert Match objects to dictionaries
    heuristic_matches = [
        {
            'algorithm': m.algorithm,
            'confidence': m.confidence,
            'reason': m.reason,
            'metadata': m.metadata
        }
        for m in heuristic_result['matches']
    ]
    
    ml_used = False
    final_matches = heuristic_matches
    
    # Phase 2: ML classification (if enabled and available)
    if use_ml:
        try:
            # Lazy load ML classifier
            if _ml_classifier is None:
                _ml_classifier = MLClassifier()
            
            # Only use ML if model is available
            if _ml_classifier.is_available():
                # Extract features (use cached instance)
                if _feature_extractor is None:
                    _feature_extractor = FeatureExtractor()
                features = _feature_extractor.extract(input_string)
                
                # Get ML predictions
                _ml_classifier.load_model()
                ml_predictions = _ml_classifier.predict(features)
                
                # Fuse heuristic and ML results
                fuser = ConfidenceFuser(
                    heuristic_weight=0.4,
                    ml_weight=0.6
                )
                final_matches = fuser.fuse(heuristic_matches, ml_predictions, context)
                ml_used = True
                
        except FileNotFoundError:
            # Model not found, fall back to heuristics only
            pass
        except Exception:
            # Any other ML error, fall back to heuristics
            pass
    
    return IdentificationResult(
        matches=final_matches,
        metadata={
            **heuristic_result['metadata'],
            'ml_enhanced': ml_used
        },
        ml_used=ml_used
    )


def identify(input_string: str, 
             context: Optional[Dict[str, Any]] = None,
             use_ml: bool = True,
             use_cache: bool = True) -> IdentificationResult:
    """
    Identify hash or format type.
    
    This is the main entry point for the hashmind library.
    Combines fast heuristic detection with optional ML classification.
    
    Args:
        input_string: String to identify
        context: Optional context for better identification
        use_ml: Whether to use ML classification (default: True, falls back if unavailable)
        use_cache: Whether to use result caching (default: True)
        
    Returns:
        IdentificationResult with matches and metadata
        
    Example:
        >>> from hashmind import identify
        >>> result = identify("5d41402abc4b2a76b9719d911017c592")
        >>> print(result.top_match())
        md5_hex
    """
    # Use cache if enabled and no context provided (context makes caching complex)
    if use_cache and context is None:
        cached_data = _cached_identify(input_string, use_ml)
        # Reconstruct IdentificationResult from cached tuple
        matches = [dict(m) for m in cached_data[0]]
        metadata = dict(cached_data[1])
        ml_used = cached_data[2]
        return IdentificationResult(matches=matches, metadata=metadata, ml_used=ml_used)
    else:
        return _identify_internal(input_string, context, use_ml)


def identify_batch(inputs: List[str], 
                   use_ml: bool = True,
                   show_progress: bool = False) -> List[IdentificationResult]:
    """
    Identify multiple hashes efficiently.
    
    Uses batch processing and caching for improved performance.
    
    Args:
        inputs: List of strings to identify
        use_ml: Whether to use ML classification
        show_progress: Show progress bar (requires tqdm)
        
    Returns:
        List of IdentificationResult objects
    """
    results = []
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(inputs, desc="Identifying")
        except ImportError:
            iterator = inputs
    else:
        iterator = inputs
    
    for input_string in iterator:
        result = identify(input_string, use_ml=use_ml, use_cache=True)
        results.append(result)
    
    return results


def clear_cache() -> None:
    """Clear the identification result cache."""
    _cached_identify.cache_clear()


def get_cache_info() -> Dict[str, int]:
    """Get cache statistics."""
    info = _cached_identify.cache_info()
    return {
        'hits': info.hits,
        'misses': info.misses,
        'size': info.currsize,
        'maxsize': info.maxsize
    }
