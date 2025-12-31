"""
Active learning module for continuous model improvement.

Implements uncertainty sampling to identify hashes where the model
needs more training data, and provides mechanisms for incremental
model updates.

Author: Supun Hewagamage (@supunhg)
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class UncertainSample:
    """Represents a sample with high uncertainty."""
    hash_value: str
    predictions: Dict[str, float]
    top_prediction: str
    confidence: float
    entropy: float
    timestamp: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FeedbackSample:
    """User-corrected sample for retraining."""
    hash_value: str
    predicted_type: str
    actual_type: str
    features: Dict[str, float]
    timestamp: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class ActiveLearner:
    """
    Active learning coordinator for model improvement.
    
    Tracks uncertain predictions and user corrections,
    enabling incremental model updates.
    """
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.6,
        feedback_file: Optional[Path] = None,
        uncertain_file: Optional[Path] = None
    ):
        """
        Initialize active learner.
        
        Args:
            uncertainty_threshold: Confidence below which to flag samples
            feedback_file: Path to store user feedback
            uncertain_file: Path to store uncertain samples
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.feedback_file = feedback_file or Path("samples/feedback.jsonl")
        self.uncertain_file = uncertain_file or Path("samples/uncertain.jsonl")
        
        # Create directories if needed
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        self.uncertain_file.parent.mkdir(parents=True, exist_ok=True)
    
    def check_uncertainty(
        self,
        hash_value: str,
        predictions: Dict[str, float]
    ) -> Optional[UncertainSample]:
        """
        Check if prediction is uncertain.
        
        Args:
            hash_value: The hash being identified
            predictions: Probability distribution over hash types
            
        Returns:
            UncertainSample if uncertain, None otherwise
        """
        if not predictions:
            return None
        
        # Get top prediction and confidence
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_type, top_conf = sorted_preds[0]
        
        # Calculate entropy (measure of uncertainty)
        import math
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in predictions.values())
        
        # Check if uncertain
        if top_conf < self.uncertainty_threshold or entropy > 2.0:
            return UncertainSample(
                hash_value=hash_value,
                predictions=predictions,
                top_prediction=top_type,
                confidence=top_conf,
                entropy=entropy,
                timestamp=datetime.now().isoformat()
            )
        
        return None
    
    def log_uncertain(self, sample: UncertainSample) -> None:
        """Log an uncertain sample for review."""
        with open(self.uncertain_file, 'a') as f:
            f.write(json.dumps(sample.to_dict()) + '\n')
    
    def add_feedback(
        self,
        hash_value: str,
        predicted_type: str,
        actual_type: str,
        features: Dict[str, float]
    ) -> None:
        """
        Add user feedback for a prediction.
        
        Args:
            hash_value: The hash that was identified
            predicted_type: What the model predicted
            actual_type: Correct hash type (user-provided)
            features: Extracted features for this sample
        """
        feedback = FeedbackSample(
            hash_value=hash_value,
            predicted_type=predicted_type,
            actual_type=actual_type,
            features=features,
            timestamp=datetime.now().isoformat()
        )
        
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback.to_dict()) + '\n')
    
    def load_feedback(self) -> List[FeedbackSample]:
        """Load all feedback samples."""
        if not self.feedback_file.exists():
            return []
        
        samples = []
        with open(self.feedback_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(FeedbackSample(**data))
        
        return samples
    
    def load_uncertain(self) -> List[UncertainSample]:
        """Load all uncertain samples."""
        if not self.uncertain_file.exists():
            return []
        
        samples = []
        with open(self.uncertain_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(UncertainSample(**data))
        
        return samples
    
    def get_training_candidates(self, limit: int = 100) -> List[UncertainSample]:
        """
        Get top uncertain samples for manual labeling.
        
        Args:
            limit: Maximum number of samples to return
            
        Returns:
            List of most uncertain samples
        """
        uncertain = self.load_uncertain()
        
        # Sort by entropy (higher = more uncertain)
        uncertain.sort(key=lambda x: x.entropy, reverse=True)
        
        return uncertain[:limit]
    
    def prepare_retraining_data(self) -> Tuple[List[Dict], List[str]]:
        """
        Prepare feedback data for model retraining.
        
        Returns:
            Tuple of (feature_dicts, labels)
        """
        feedback = self.load_feedback()
        
        if not feedback:
            return [], []
        
        features = [sample.features for sample in feedback]
        labels = [sample.actual_type for sample in feedback]
        
        return features, labels
    
    def retrain_model(
        self,
        classifier,
        feature_extractor,
        combine_with_original: bool = True
    ) -> Dict[str, Any]:
        """
        Retrain model with feedback data.
        
        Args:
            classifier: MLClassifier instance to retrain
            feature_extractor: FeatureExtractor instance
            combine_with_original: If True, combine feedback with original training data
            
        Returns:
            Dictionary with retraining statistics
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # Load feedback
        features, labels = self.prepare_retraining_data()
        
        if not features:
            return {"status": "no_feedback", "samples_added": 0}
        
        # Convert to DataFrame
        df_feedback = pd.DataFrame(features)
        
        # If combining with original data
        if combine_with_original and classifier.model is not None:
            # Load original training data if available
            original_data_path = Path("samples/training_data.jsonl")
            if original_data_path.exists():
                import json
                original_samples = []
                with open(original_data_path, 'r') as f:
                    for line in f:
                        original_samples.append(json.loads(line.strip()))
                
                # Extract features from original samples
                original_features = []
                original_labels = []
                for sample in original_samples[:1000]:  # Limit to avoid memory issues
                    feats = feature_extractor.extract(sample['hash'])
                    if feats:
                        original_features.append(feats)
                        original_labels.append(sample['type'])
                
                # Combine
                df_original = pd.DataFrame(original_features)
                df_combined = pd.concat([df_original, df_feedback], ignore_index=True)
                labels = original_labels + labels
            else:
                df_combined = df_feedback
        else:
            df_combined = df_feedback
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df_combined, labels, test_size=0.2, random_state=42
        )
        
        # Retrain
        classifier.train(X_train, y_train)
        
        # Evaluate
        from sklearn.metrics import accuracy_score
        y_pred = classifier.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save updated model
        classifier.save()
        
        return {
            "status": "success",
            "samples_added": len(features),
            "total_samples": len(df_combined),
            "test_accuracy": accuracy
        }
    
    def clear_uncertain(self) -> int:
        """
        Clear uncertain samples log.
        
        Returns:
            Number of samples cleared
        """
        count = 0
        if self.uncertain_file.exists():
            with open(self.uncertain_file, 'r') as f:
                count = sum(1 for _ in f)
            self.uncertain_file.unlink()
        
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get active learning statistics."""
        feedback = self.load_feedback()
        uncertain = self.load_uncertain()
        
        # Count corrections by type
        corrections = {}
        for sample in feedback:
            key = f"{sample.predicted_type} → {sample.actual_type}"
            corrections[key] = corrections.get(key, 0) + 1
        
        return {
            "total_feedback": len(feedback),
            "total_uncertain": len(uncertain),
            "correction_patterns": corrections,
            "avg_uncertainty": sum(s.entropy for s in uncertain) / len(uncertain) if uncertain else 0
        }


def create_active_learner(
    uncertainty_threshold: float = 0.6,
    feedback_dir: Optional[Path] = None
) -> ActiveLearner:
    """
    Factory function to create an ActiveLearner.
    
    Args:
        uncertainty_threshold: Confidence threshold for flagging samples
        feedback_dir: Directory to store feedback files
        
    Returns:
        Configured ActiveLearner instance
    """
    if feedback_dir:
        feedback_file = feedback_dir / "feedback.jsonl"
        uncertain_file = feedback_dir / "uncertain.jsonl"
    else:
        feedback_file = None
        uncertain_file = None
    
    return ActiveLearner(
        uncertainty_threshold=uncertainty_threshold,
        feedback_file=feedback_file,
        uncertain_file=uncertain_file
    )
