"""
Adaptive confidence threshold tuning.

Dynamically adjusts confidence thresholds based on hash type,
detection method, and historical accuracy.

Author: Supun Hewagamage (@supunhg)
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ThresholdConfig:
    """Configuration for confidence thresholds."""
    # Minimum confidence to report a match
    min_confidence: float = 0.5
    
    # Thresholds by hash type (for known problematic types)
    type_thresholds: Dict[str, float] = None
    
    # Thresholds by detection method
    method_thresholds: Dict[str, float] = None
    
    # Dynamic adjustment enabled
    adaptive: bool = True
    
    def __post_init__(self):
        if self.type_thresholds is None:
            self.type_thresholds = {}
        if self.method_thresholds is None:
            self.method_thresholds = {
                'length_only': 0.3,  # Length-based detection is less reliable
                'prefix': 0.7,       # Prefix detection is quite reliable
                'regex': 0.8,        # Regex detection is very reliable
                'charset': 0.4,      # Character set alone is weak
                'entropy': 0.5,      # Entropy is moderate
                'ml': 0.6,           # ML predictions need decent confidence
            }


class ThresholdTuner:
    """
    Manages adaptive confidence thresholds.
    
    Tracks detection accuracy and adjusts thresholds to optimize
    precision/recall tradeoff.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize threshold tuner.
        
        Args:
            config_file: Path to threshold configuration file
        """
        self.config_file = config_file or Path("models/thresholds.json")
        self.config = self._load_config()
        
        # Track accuracy by hash type and method
        self.accuracy_history: Dict[str, list] = {}
    
    def _load_config(self) -> ThresholdConfig:
        """Load threshold configuration from file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                return ThresholdConfig(**data)
        else:
            return ThresholdConfig()
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def get_threshold(
        self,
        hash_type: Optional[str] = None,
        method: Optional[str] = None
    ) -> float:
        """
        Get confidence threshold for given context.
        
        Args:
            hash_type: Type of hash (e.g., 'md5', 'sha256')
            method: Detection method used
            
        Returns:
            Confidence threshold to use
        """
        # Start with global minimum
        threshold = self.config.min_confidence
        
        # Adjust for hash type
        if hash_type and hash_type in self.config.type_thresholds:
            threshold = max(threshold, self.config.type_thresholds[hash_type])
        
        # Adjust for detection method
        if method and method in self.config.method_thresholds:
            threshold = max(threshold, self.config.method_thresholds[method])
        
        return threshold
    
    def should_report(
        self,
        confidence: float,
        hash_type: Optional[str] = None,
        method: Optional[str] = None
    ) -> bool:
        """
        Determine if a match should be reported.
        
        Args:
            confidence: Confidence score
            hash_type: Type of hash detected
            method: Detection method used
            
        Returns:
            True if confidence exceeds threshold
        """
        threshold = self.get_threshold(hash_type, method)
        return confidence >= threshold
    
    def record_accuracy(
        self,
        hash_type: str,
        method: str,
        was_correct: bool
    ) -> None:
        """
        Record detection accuracy for adaptive tuning.
        
        Args:
            hash_type: Type that was detected
            method: Method that detected it
            was_correct: Whether detection was correct
        """
        key = f"{hash_type}:{method}"
        
        if key not in self.accuracy_history:
            self.accuracy_history[key] = []
        
        self.accuracy_history[key].append(1.0 if was_correct else 0.0)
        
        # Keep only recent history (last 100 samples)
        if len(self.accuracy_history[key]) > 100:
            self.accuracy_history[key] = self.accuracy_history[key][-100:]
        
        # Update thresholds if adaptive mode enabled
        if self.config.adaptive:
            self._update_thresholds(hash_type, method)
    
    def _update_thresholds(self, hash_type: str, method: str) -> None:
        """Update thresholds based on accuracy history."""
        key = f"{hash_type}:{method}"
        
        if key not in self.accuracy_history or len(self.accuracy_history[key]) < 10:
            return
        
        # Calculate recent accuracy
        recent = self.accuracy_history[key][-20:]
        accuracy = sum(recent) / len(recent)
        
        # Adjust threshold for this hash type
        if accuracy < 0.7:
            # Low accuracy - increase threshold to be more conservative
            current = self.config.type_thresholds.get(hash_type, self.config.min_confidence)
            new_threshold = min(current + 0.05, 0.95)
            self.config.type_thresholds[hash_type] = new_threshold
        elif accuracy > 0.95:
            # Very high accuracy - can lower threshold to catch more
            current = self.config.type_thresholds.get(hash_type, self.config.min_confidence)
            new_threshold = max(current - 0.05, self.config.min_confidence)
            self.config.type_thresholds[hash_type] = new_threshold
        
        # Save updated config
        self.save_config()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get accuracy statistics."""
        stats = {}
        
        for key, history in self.accuracy_history.items():
            if len(history) >= 5:
                accuracy = sum(history) / len(history)
                stats[key] = {
                    'accuracy': accuracy,
                    'samples': len(history),
                    'threshold': self.get_threshold(
                        hash_type=key.split(':')[0],
                        method=key.split(':')[1]
                    )
                }
        
        return stats
    
    def reset_thresholds(self) -> None:
        """Reset all thresholds to defaults."""
        self.config = ThresholdConfig()
        self.accuracy_history.clear()
        self.save_config()
    
    def set_type_threshold(self, hash_type: str, threshold: float) -> None:
        """
        Manually set threshold for a hash type.
        
        Args:
            hash_type: Hash type identifier
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.config.type_thresholds[hash_type] = threshold
        self.save_config()
    
    def set_method_threshold(self, method: str, threshold: float) -> None:
        """
        Manually set threshold for a detection method.
        
        Args:
            method: Detection method name
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.config.method_thresholds[method] = threshold
        self.save_config()
    
    def optimize_for_precision(self) -> None:
        """Optimize thresholds for high precision (fewer false positives)."""
        # Increase all thresholds
        self.config.min_confidence = 0.7
        for method in self.config.method_thresholds:
            self.config.method_thresholds[method] *= 1.2
            self.config.method_thresholds[method] = min(
                self.config.method_thresholds[method], 0.95
            )
        self.save_config()
    
    def optimize_for_recall(self) -> None:
        """Optimize thresholds for high recall (catch more matches)."""
        # Decrease all thresholds
        self.config.min_confidence = 0.3
        for method in self.config.method_thresholds:
            self.config.method_thresholds[method] *= 0.8
            self.config.method_thresholds[method] = max(
                self.config.method_thresholds[method], 0.2
            )
        self.save_config()


# Global tuner instance
_global_tuner: Optional[ThresholdTuner] = None


def get_tuner() -> ThresholdTuner:
    """Get or create global threshold tuner instance."""
    global _global_tuner
    if _global_tuner is None:
        _global_tuner = ThresholdTuner()
    return _global_tuner


def should_report_match(
    confidence: float,
    hash_type: Optional[str] = None,
    method: Optional[str] = None
) -> bool:
    """
    Convenience function to check if match should be reported.
    
    Args:
        confidence: Match confidence score
        hash_type: Detected hash type
        method: Detection method used
        
    Returns:
        True if match meets threshold criteria
    """
    tuner = get_tuner()
    return tuner.should_report(confidence, hash_type, method)
