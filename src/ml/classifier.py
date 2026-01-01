"""ML classifier for hash identification."""

from typing import Dict, List, Any, Optional
import os
import pickle


class MLClassifier:
    """
    Machine learning classifier for hash identification.
    
    Uses XGBoost model trained on synthetic hash data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML classifier.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.model_path = model_path or self._get_default_model_path()
        self._loaded = False
        
    def _get_default_model_path(self) -> str:
        """Get default model path."""
        # Look for model in models/ directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(project_root, 'models', 'hashmind_model.pkl')
    
    def load_model(self):
        """Load the trained model from disk."""
        if self._loaded:
            return
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Train a model first using: python scripts/train_model.py"
            )
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self._loaded = True
    
    def predict(self, features: Dict[str, Any]) -> List[Dict[str, float]]:
        """
        Predict hash type from features.
        
        Args:
            features: Feature dictionary from FeatureExtractor
            
        Returns:
            List of predictions with probabilities, sorted by confidence
        """
        if not self._loaded:
            self.load_model()
        
        import pandas as pd
        import numpy as np
        
        # Convert features to DataFrame with correct column order
        df = pd.DataFrame([features])
        
        # Ensure all expected features are present
        for fname in self.feature_names:
            if fname not in df.columns:
                df[fname] = 0
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Convert boolean and categorical features
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
            elif df[col].dtype == object:
                df[col] = pd.Categorical(df[col]).codes
        
        # Predict probabilities
        proba = self.model.predict_proba(df.values)[0]
        
        # Create result list
        results = []
        for idx, prob in enumerate(proba):
            if prob > 0.01:  # Only include predictions > 1%
                results.append({
                    'algorithm': self.label_encoder.classes_[idx],
                    'probability': float(prob)
                })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return results
    
    def predict_batch(self, features_batch: List[Dict[str, Any]]) -> List[List[Dict[str, float]]]:
        """
        Batch prediction for efficiency.
        
        Args:
            features_batch: List of feature dictionaries
            
        Returns:
            List of prediction lists
        """
        if not self._loaded:
            self.load_model()
        
        if not features_batch:
            return []
        
        import pandas as pd
        import numpy as np
        
        # Convert all features to DataFrame at once (much faster)
        df = pd.DataFrame(features_batch)
        
        # Ensure all expected features are present
        for fname in self.feature_names:
            if fname not in df.columns:
                df[fname] = 0
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Convert boolean and categorical features
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
            elif df[col].dtype == object:
                df[col] = pd.Categorical(df[col]).codes
        
        # Predict all at once
        proba_batch = self.model.predict_proba(df.values)
        
        # Create result lists
        all_results = []
        for proba in proba_batch:
            results = []
            for idx, prob in enumerate(proba):
                if prob > 0.01:  # Only include predictions > 1%
                    results.append({
                        'algorithm': self.label_encoder.classes_[idx],
                        'probability': float(prob)
                    })
            results.sort(key=lambda x: x['probability'], reverse=True)
            all_results.append(results)
        
        return all_results
    
    def is_available(self) -> bool:
        """Check if model is available."""
        return os.path.exists(self.model_path)
