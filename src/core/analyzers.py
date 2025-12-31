"""Statistical and structural analyzers for hash detection."""

import math
from collections import Counter
from typing import Dict, Set, List
from .matchers import Match


class CharacterSetAnalyzer:
    """Analyze character set distribution."""
    
    @staticmethod
    def analyze(input_string: str) -> Dict[str, any]:
        """
        Analyze character set composition.
        
        Args:
            input_string: Input string to analyze
            
        Returns:
            Dictionary with character set analysis
        """
        chars = set(input_string)
        
        hex_chars = set('0123456789abcdefABCDEF')
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        base32_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ234567=')
        alpha_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        digit_chars = set('0123456789')
        
        return {
            'is_hex': chars.issubset(hex_chars) and len(chars) > 0,
            'is_base64': chars.issubset(base64_chars),
            'is_base32': chars.issubset(base32_chars),
            'is_alphanumeric': chars.issubset(alpha_chars | digit_chars),
            'has_uppercase': any(c.isupper() for c in input_string),
            'has_lowercase': any(c.islower() for c in input_string),
            'has_digits': any(c.isdigit() for c in input_string),
            'has_special': not chars.issubset(alpha_chars | digit_chars),
            'unique_chars': len(chars),
            'char_diversity': len(chars) / len(input_string) if input_string else 0,
        }
    
    def get_charset_matches(self, input_string: str) -> List[Match]:
        """
        Generate matches based on character set analysis.
        
        Args:
            input_string: Input string to analyze
            
        Returns:
            List of charset-based matches
        """
        analysis = self.analyze(input_string)
        matches = []
        
        # Don't suggest hex encoding if string starts with known prefixes
        known_prefixes = ('$', '{', '0x', 'sha1$', 'md5$')
        has_known_prefix = any(input_string.startswith(p) for p in known_prefixes)
        
        if analysis['is_hex'] and not has_known_prefix:
            matches.append(Match(
                algorithm='hex_encoding',
                confidence=0.4,  # Lower confidence - hex is common, not specific
                reason="String contains only hexadecimal characters",
                metadata=analysis
            ))
        
        # Only suggest base64 for strings without special format prefixes
        if analysis['is_base64'] and '=' in input_string[-2:] and not has_known_prefix:
            # Don't suggest base64 if it looks like part of a structured format
            if '.' not in input_string:  # Avoid JWT confusion
                matches.append(Match(
                    algorithm='base64_encoding',
                    confidence=0.6,
                    reason="Matches base64 character set with padding",
                    metadata=analysis
                ))
        
        if analysis['is_base32'] and len(input_string) % 8 == 0:
            matches.append(Match(
                algorithm='base32_encoding',
                confidence=0.5,
                reason="Matches base32 character set with proper length",
                metadata=analysis
            ))
        
        return matches


class StructureValidator:
    """Validate format-specific structures."""
    
    def validate_bcrypt(self, input_string: str) -> bool:
        """Validate bcrypt structure."""
        # $2a$10$saltsaltsaltsalthashhashhashhashhashhashhash
        if not input_string.startswith(('$2a$', '$2b$', '$2y$')):
            return False
        
        parts = input_string.split('$')
        if len(parts) != 4:
            return False
        
        # Check cost parameter (usually 04-31)
        try:
            cost = int(parts[2])
            if not (4 <= cost <= 31):
                return False
        except ValueError:
            return False
        
        # Check salt + hash length (22 + 31 = 53 chars in base64)
        return len(parts[3]) == 53
    
    def validate_jwt(self, input_string: str) -> Dict[str, any]:
        """
        Validate JWT structure and extract metadata.
        
        Args:
            input_string: Potential JWT string
            
        Returns:
            Validation result with metadata
        """
        parts = input_string.split('.')
        if len(parts) != 3:
            return {'valid': False}
        
        # All parts should be base64url encoded
        import re
        base64url_pattern = re.compile(r'^[A-Za-z0-9_-]+$')
        
        if not all(base64url_pattern.match(part) for part in parts):
            return {'valid': False}
        
        return {
            'valid': True,
            'header_length': len(parts[0]),
            'payload_length': len(parts[1]),
            'signature_length': len(parts[2]),
        }


class EntropyCalculator:
    """Calculate various entropy measures."""
    
    @staticmethod
    def shannon_entropy(input_string: str) -> float:
        """
        Calculate Shannon entropy.
        
        Args:
            input_string: Input string
            
        Returns:
            Shannon entropy value (0-8 for byte entropy)
        """
        if not input_string:
            return 0.0
        
        # Count character frequencies
        counter = Counter(input_string)
        length = len(input_string)
        
        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    @staticmethod
    def compression_ratio(input_string: str) -> float:
        """
        Estimate entropy via compression ratio.
        
        Args:
            input_string: Input string
            
        Returns:
            Compression ratio (compressed/original)
        """
        import zlib
        
        try:
            original_size = len(input_string.encode('utf-8'))
            compressed_size = len(zlib.compress(input_string.encode('utf-8'), level=9))
            return compressed_size / original_size if original_size > 0 else 1.0
        except Exception:
            return 1.0
    
    def analyze_entropy(self, input_string: str) -> Dict[str, float]:
        """
        Comprehensive entropy analysis.
        
        Args:
            input_string: Input string
            
        Returns:
            Dictionary with various entropy measures
        """
        return {
            'shannon': self.shannon_entropy(input_string),
            'compression_ratio': self.compression_ratio(input_string),
            'normalized_shannon': self.shannon_entropy(input_string) / 8.0,  # Normalize to 0-1
        }
