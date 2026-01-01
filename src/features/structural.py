"""Feature extraction for ML-based classification (Phase 2)."""

from typing import Dict, List, Any
import math
import re
import zlib
from collections import Counter
import functools

# Pre-compiled regex patterns for performance
BASE64_PATTERN = re.compile(r'^[A-Za-z0-9+/]+=*$')
HEX_PATTERN = re.compile(r'^[0-9a-fA-F]+$')
BASE32_PATTERN = re.compile(r'^[A-Z2-7]+=*$')
BASE64URL_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
JWT_PATTERN = re.compile(r'^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$')


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
        """Extract statistical features with optimized calculations."""
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
        
        # Single pass for character counting
        byte_values = []
        digit_count = 0
        alpha_count = 0
        counter = {}
        
        for c in input_string:
            bval = ord(c)
            byte_values.append(bval)
            counter[c] = counter.get(c, 0) + 1
            if c.isdigit():
                digit_count += 1
            elif c.isalpha():
                alpha_count += 1
        
        # Statistical calculations
        mean_byte = sum(byte_values) / length
        variance = sum((b - mean_byte) ** 2 for b in byte_values) / length
        special_count = length - digit_count - alpha_count
        
        # Entropy calculation (use cached)
        entropy = _calculate_entropy(input_string)
        
        # Character frequency stats
        freq_values = list(counter.values())
        max_freq = max(freq_values)
        min_freq = min(freq_values)
        
        return {
            'entropy': entropy,
            'normalized_entropy': entropy / 8.0,
            'mean_byte_value': mean_byte,
            'variance': variance,
            'std_dev': variance ** 0.5,  # Faster than math.sqrt
            'unique_chars': len(counter),
            'unique_ratio': len(counter) / length,
            'max_char_freq': max_freq / length,
            'min_char_freq': min_freq / length,
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
        # Use compiled patterns for faster matching
        return {
            'is_valid_base64': _is_valid_base64(input_string),
            'is_valid_base32': _is_valid_base32(input_string),
            'is_valid_hex': _is_valid_hex(input_string),
            'is_valid_base64url': _is_valid_base64url(input_string),
            'has_padding': input_string.endswith('='),
            'padding_count': len(input_string) - len(input_string.rstrip('=')),
            'padding_ratio': (len(input_string) - len(input_string.rstrip('='))) / len(input_string) if input_string else 0,
            'compression_ratio': _compression_ratio(input_string),
            'matches_uuid_pattern': bool(UUID_PATTERN.match(input_string)),
            'matches_jwt_pattern': bool(JWT_PATTERN.match(input_string)),
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


@functools.lru_cache(maxsize=2048)
def _calculate_entropy(s: str) -> float:
    """Calculate Shannon entropy (cached)."""
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
    return bool(BASE64_PATTERN.match(s))


def _is_valid_hex(s: str) -> bool:
    """Check if string is valid hex."""
    return bool(HEX_PATTERN.match(s))


def _is_valid_base32(s: str) -> bool:
    """Check if string is valid base32."""
    return bool(BASE32_PATTERN.match(s))


def _is_valid_base64url(s: str) -> bool:
    """Check if string is valid base64url (JWT variant)."""
    return bool(BASE64URL_PATTERN.match(s))


def _bigram_diversity(s: str) -> float:
    """Calculate bigram diversity."""
    if len(s) < 2:
        return 0.0
    bigrams = [s[i:i+2] for i in range(len(s)-1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


@functools.lru_cache(maxsize=1024)
def _compression_ratio(s: str) -> float:
    """Calculate compression ratio using zlib (cached)."""
    try:
        original_size = len(s.encode('utf-8'))
        if original_size == 0:
            return 1.0
        compressed_size = len(zlib.compress(s.encode('utf-8'), level=6))  # Level 6 is faster, good enough
        return compressed_size / original_size
    except Exception:
        return 1.0
