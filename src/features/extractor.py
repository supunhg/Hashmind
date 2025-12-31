"""Main feature extraction engine."""

from typing import Dict, Any
from .structural import StructuralFeatures, StatisticalFeatures, AlgorithmicFeatures


class FeatureExtractor:
    """
    Main feature extraction engine combining all feature types.
    
    This will be used in Phase 2 for ML-based classification.
    """
    
    def __init__(self):
        """Initialize feature extractors."""
        self.structural = StructuralFeatures()
        self.statistical = StatisticalFeatures()
        self.algorithmic = AlgorithmicFeatures()
    
    def extract(self, input_string: str) -> Dict[str, Any]:
        """
        Extract all features from input string.
        
        Args:
            input_string: String to extract features from
            
        Returns:
            Dictionary containing all feature types
        """
        features = {}
        
        # Extract each feature type
        features.update({
            f"struct_{k}": v 
            for k, v in self.structural.extract(input_string).items()
        })
        
        features.update({
            f"stat_{k}": v 
            for k, v in self.statistical.extract(input_string).items()
        })
        
        features.update({
            f"algo_{k}": v 
            for k, v in self.algorithmic.extract(input_string).items()
        })
        
        return features
    
    def extract_batch(self, input_strings: list) -> list:
        """
        Extract features for multiple strings.
        
        Args:
            input_strings: List of strings
            
        Returns:
            List of feature dictionaries
        """
        return [self.extract(s) for s in input_strings]
