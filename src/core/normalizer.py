"""Input normalization and preprocessing."""

import re
from typing import Optional


class InputNormalizer:
    """Normalize and validate input strings."""
    
    def __init__(self):
        """Initialize the normalizer."""
        self.max_length = 1_000_000  # 1MB text limit
    
    def normalize(self, input_string: str) -> str:
        """
        Normalize input string.
        
        Args:
            input_string: Raw input string
            
        Returns:
            Normalized string
            
        Raises:
            ValueError: If input is invalid or too long
        """
        if not isinstance(input_string, str):
            raise ValueError("Input must be a string")
        
        if len(input_string) > self.max_length:
            raise ValueError(f"Input exceeds maximum length of {self.max_length}")
        
        # Strip leading/trailing whitespace
        normalized = input_string.strip()
        
        if not normalized:
            raise ValueError("Input cannot be empty")
        
        return normalized
    
    def detect_encoding(self, data: bytes) -> str:
        """
        Detect character encoding of byte data.
        
        Args:
            data: Raw bytes
            
        Returns:
            Detected encoding name
        """
        # Simple UTF-8 detection for now
        try:
            data.decode('utf-8')
            return 'utf-8'
        except UnicodeDecodeError:
            return 'unknown'
    
    def clean_whitespace(self, text: str) -> str:
        """
        Remove or normalize internal whitespace.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        return re.sub(r'\s+', ' ', text)
