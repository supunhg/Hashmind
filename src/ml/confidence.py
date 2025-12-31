"""Confidence fusion for combining heuristic and ML results."""

from typing import List, Dict, Any
from collections import defaultdict


class ConfidenceFuser:
    """
    Fuse confidence scores from multiple detection methods.
    
    Combines:
    - Heuristic detection scores
    - ML prediction probabilities  
    - Context-based adjustments
    """
    
    def __init__(self, 
                 heuristic_weight: float = 0.4,
                 ml_weight: float = 0.6,
                 context_weight: float = 0.0):
        """
        Initialize confidence fuser.
        
        Args:
            heuristic_weight: Weight for heuristic scores (default: 0.4)
            ml_weight: Weight for ML predictions (default: 0.6)
            context_weight: Weight for contextual information (default: 0.0)
        """
        self.heuristic_weight = heuristic_weight
        self.ml_weight = ml_weight
        self.context_weight = context_weight
        
        # Normalize weights
        total = heuristic_weight + ml_weight + context_weight
        if total > 0:
            self.heuristic_weight /= total
            self.ml_weight /= total
            self.context_weight /= total
    
    def fuse(self,
             heuristic_matches: List[Dict[str, Any]],
             ml_predictions: List[Dict[str, float]] = None,
             context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple sources.
        
        Args:
            heuristic_matches: Matches from heuristic detection
            ml_predictions: Predictions from ML classifier (optional)
            context: Optional contextual information
            
        Returns:
            Fused and ranked results
        """
        # If no ML predictions, return heuristics only
        if not ml_predictions:
            return heuristic_matches
        
        # Create a dictionary to accumulate scores
        fused_scores = defaultdict(lambda: {
            'heuristic_score': 0.0,
            'ml_score': 0.0,
            'reason': '',
            'metadata': {}
        })
        
        # Add heuristic scores
        for match in heuristic_matches:
            algo = match['algorithm']
            fused_scores[algo]['heuristic_score'] = match['confidence']
            fused_scores[algo]['reason'] = match.get('reason', '')
            fused_scores[algo]['metadata'] = match.get('metadata', {})
        
        # Add ML scores
        for pred in ml_predictions:
            algo = pred['algorithm']
            fused_scores[algo]['ml_score'] = pred['probability']
            
            # If we don't have a heuristic reason, use ML
            if not fused_scores[algo]['reason']:
                fused_scores[algo]['reason'] = f"ML prediction ({pred['probability']:.1%} confidence)"
        
        # Calculate fused confidence
        results = []
        for algo, scores in fused_scores.items():
            fused_confidence = (
                self.heuristic_weight * scores['heuristic_score'] +
                self.ml_weight * scores['ml_score']
            )
            
            # Build combined reason
            reason_parts = []
            if scores['heuristic_score'] > 0:
                reason_parts.append(scores['reason'])
            if scores['ml_score'] > 0:
                reason_parts.append(f"ML: {scores['ml_score']:.1%}")
            
            results.append({
                'algorithm': algo,
                'confidence': fused_confidence,
                'reason': '; '.join(reason_parts) if reason_parts else 'Combined detection',
                'metadata': {
                    **scores['metadata'],
                    'heuristic_confidence': scores['heuristic_score'],
                    'ml_confidence': scores['ml_score'],
                    'fusion_weights': {
                        'heuristic': self.heuristic_weight,
                        'ml': self.ml_weight,
                    }
                }
            })
        
        # Sort by fused confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
