"""Feature extraction for ML-based classification (Phase 2)."""

from typing import Dict, List, Any
import math
from collections import Counter


class StructuralFeatures:
    """Extract structural features from input strings."""
    
    @staticmethod
    def extract(input_string: str) -> Dict[str, Any]:
        """Extract structural features."""
        length = len(input_string)
        
        return {
            'length': length,
            'length_category': _categorize_length(length),
            'length_mod_8': length % 8,
            'length_mod_4': length % 4,
            'has_dots': '.' in input_string,
            'has_dashes': '-' in input_string,
            'has_slashes': '/' in input_string,
            'has_dollar': '$' in input_string,
            'has_colon': ':' in input_string,
            'has_equals': '=' in input_string,
            'dot_count': input_string.count('.'),
            'dash_count': input_string.count('-'),
            'slash_count': input_string.count('/'),
            'dollar_count': input_string.count('$'),
            'equals_count': input_string.count('='),
            'block_count': len(input_string.split('$')) if '$' in input_string else 1,
            'part_count_dot': len(input_string.split('.')),
            'part_count_dash': len(input_string.split('-')),
            'has_uppercase': any(c.isupper() for c in input_string),
            'has_lowercase': any(c.islower() for c in input_string),
            'is_all_upper': input_string.isupper() if input_string.isalpha() else False,
            'is_all_lower': input_string.islower() if input_string.isalpha() else False,
            'uppercase_ratio': sum(1 for c in input_string if c.isupper()) / length if length else 0,
            'starts_with_digit': input_string[0].isdigit() if input_string else False,
            'starts_with_letter': input_string[0].isalpha() if input_string else False,
            'starts_with_special': not input_string[0].isalnum() if input_string else False,
            'ends_with_equals': input_string.endswith('='),
            'ends_with_digit': input_string[-1].isdigit() if input_string else False,
        }


class StatisticalFeatures:
    """Extract statistical features for ML."""
    
    @staticmethod
    def extract(input_string: str) -> Dict[str, float]:
        """Extract statistical features."""
        if not input_string:
            return {
                'entropy': 0.0,
                'mean_byte_value': 0.0,
                'unique_ratio': 0.0,
                'variance': 0.0,
                'digit_ratio': 0.0,
                'alpha_ratio': 0.0,
                'special_ratio': 0.0,
            }
        
        length = len(input_string)
        byte_values = [ord(c) for c in input_string]
        counter = Counter(input_string)
        
        mean_byte = sum(byte_values) / length
        variance = sum((b - mean_byte) ** 2 for b in byte_values) / length
        
        digit_count = sum(1 for c in input_string if c.isdigit())
        alpha_count = sum(1 for c in input_string if c.isalpha())
        special_count = length - digit_count - alpha_count
        
        return {
            'entropy': _calculate_entropy(input_string),
            'normalized_entropy': _calculate_entropy(input_string) / 8.0 if length else 0,
            'mean_byte_value': mean_byte,
            'variance': variance,
            'std_dev': math.sqrt(variance),
            'unique_chars': len(counter),
            'unique_ratio': len(counter) / length,
            'max_char_freq': max(counter.values()) / length,
            'min_char_freq': min(counter.values()) / length,
            'digit_ratio': digit_count / length,
            'alpha_ratio': alpha_count / length,
            'special_ratio': special_count / length,
            'bigram_diversity': _bigram_diversity(input_string),
        }


class AlgorithmicFeatures:
    """Extract algorithm-specific features."""
    
    @staticmethod
    def extract(input_string: str) -> Dict[str, Any]:
        """Extract algorithmic features."""
        import re
        
        return {
            'is_valid_base64': _is_valid_base64(input_string),
            'is_valid_base32': _is_valid_base32(input_string),
            'is_valid_hex': _is_valid_hex(input_string),
            'is_valid_base64url': _is_valid_base64url(input_string),
            'has_padding': input_string.endswith('='),
            'padding_count': len(input_string) - len(input_string.rstrip('=')),
            'padding_ratio': (len(input_string) - len(input_string.rstrip('='))) / len(input_string) if input_string else 0,
            'compression_ratio': _compression_ratio(input_string),
            'matches_uuid_pattern': bool(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', input_string, re.I)),
            'matches_jwt_pattern': bool(re.match(r'^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$', input_string)),
            'matches_bcrypt_pattern': input_string.startswith(('$2a$', '$2b$', '$2y$')),
            'matches_crypt_pattern': input_string.startswith('$') and input_string.count('$') >= 2,
            'has_hash_prefix': input_string.startswith(('$', '{', '0x')),
            'has_brackets': '{' in input_string or '[' in input_string,
        }


# Helper functions

def _categorize_length(length: int) -> str:
    """Categorize length into buckets."""
    if length <= 15:
        return '0-15'
    elif length <= 31:
        return '16-31'
    elif length <= 63:
        return '32-63'
    elif length <= 127:
        return '64-127'
    else:
        return '128+'


def _calculate_entropy(s: str) -> float:
    """Calculate Shannon entropy."""
    if not s:
        return 0.0
    
    counter = Counter(s)
    length = len(s)
    
    entropy = 0.0
    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    
    return entropy


def _is_valid_base64(s: str) -> bool:
    """Check if string is valid base64."""
    import re
    return bool(re.match(r'^[A-Za-z0-9+/]+=*$', s))


def _is_valid_hex(s: str) -> bool:
    """Check if string is valid hex."""
    import re
    return bool(re.match(r'^[0-9a-fA-F]+$', s))


def _is_valid_base32(s: str) -> bool:
    """Check if string is valid base32."""
    import re
    return bool(re.match(r'^[A-Z2-7]+=*$', s))


def _is_valid_base64url(s: str) -> bool:
    """Check if string is valid base64url (JWT variant)."""
    import re
    return bool(re.match(r'^[A-Za-z0-9_-]+$', s))


def _bigram_diversity(s: str) -> float:
    """Calculate bigram diversity."""
    if len(s) < 2:
        return 0.0
    bigrams = [s[i:i+2] for i in range(len(s)-1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def _compression_ratio(s: str) -> float:
    """Calculate compression ratio using zlib."""
    import zlib
    try:
        original_size = len(s.encode('utf-8'))
        if original_size == 0:
            return 1.0
        compressed_size = len(zlib.compress(s.encode('utf-8'), level=9))
        return compressed_size / original_size
    except Exception:
        return 1.0
