"""
Recursive hash decoder for handling encoded hash chains.

Detects and decodes common encoding patterns like base64(md5(x)),
hex(sha256(x)), url_encoded(bcrypt(x)), etc.

Author: Supun Hewagamage (@supunhg)
"""

import base64
import binascii
import urllib.parse
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class DecodingStep:
    """Represents a single decoding step in the chain."""
    encoding: str
    input_value: str
    output_value: str
    success: bool
    confidence: float = 1.0


@dataclass
class DecodingResult:
    """Result of recursive decoding attempt."""
    original: str
    final_value: str
    steps: List[DecodingStep]
    total_depth: int
    success: bool
    
    def get_chain(self) -> str:
        """Get human-readable decoding chain."""
        if not self.steps:
            return "no encoding"
        
        encodings = [step.encoding for step in self.steps if step.success]
        return " → ".join(encodings) if encodings else "failed"


class RecursiveDecoder:
    """
    Recursively decode encoded hashes.
    
    Supports common encodings:
    - Base64 (standard, URL-safe)
    - Hex encoding
    - URL encoding
    - Quoted-printable
    - Unicode escape sequences
    """
    
    def __init__(self, max_depth: int = 5):
        """
        Initialize decoder.
        
        Args:
            max_depth: Maximum recursion depth to prevent infinite loops
        """
        self.max_depth = max_depth
        self.decoders = {
            'base64': self._decode_base64,
            'base64url': self._decode_base64url,
            'hex': self._decode_hex,
            'url': self._decode_url,
            'quoted_printable': self._decode_quoted_printable,
            'unicode_escape': self._decode_unicode_escape,
        }
    
    def decode(self, value: str) -> DecodingResult:
        """
        Recursively decode a value.
        
        Args:
            value: Input string (potentially encoded hash)
            
        Returns:
            DecodingResult with decoding chain and final value
        """
        steps = []
        current = value
        depth = 0
        
        while depth < self.max_depth:
            # Try each decoder
            decoded, encoding = self._try_decoders(current)
            
            if decoded is None or decoded == current:
                # No more decoding possible
                break
            
            # Record successful decoding step
            steps.append(DecodingStep(
                encoding=encoding,
                input_value=current,
                output_value=decoded,
                success=True,
                confidence=self._estimate_confidence(current, decoded, encoding)
            ))
            
            current = decoded
            depth += 1
        
        return DecodingResult(
            original=value,
            final_value=current,
            steps=steps,
            total_depth=depth,
            success=depth > 0
        )
    
    def _try_decoders(self, value: str) -> Tuple[Optional[str], Optional[str]]:
        """Try all decoders and return first successful result."""
        for encoding, decoder_func in self.decoders.items():
            try:
                decoded = decoder_func(value)
                if decoded and decoded != value and self._is_valid_decoded(decoded):
                    return decoded, encoding
            except Exception:
                continue
        
        return None, None
    
    def _decode_base64(self, value: str) -> Optional[str]:
        """Decode standard Base64."""
        try:
            # Must look like base64 (alphanumeric + + / =)
            if not re.match(r'^[A-Za-z0-9+/]+=*$', value):
                return None
            
            # Need minimum length
            if len(value) < 4:
                return None
            
            # Base64 should have proper padding (length multiple of 4)
            if len(value) % 4 not in [0, 2, 3]:  # After padding, should be multiple of 4
                return None
            
            # If it's all hex characters, it's probably a hash, not base64
            if all(c in '0123456789abcdefABCDEF' for c in value):
                # Check if it's a known hash length
                if len(value) in [32, 40, 56, 64, 96, 128]:
                    return None
            
            # Add padding if missing
            missing_padding = len(value) % 4
            if missing_padding:
                value += '=' * (4 - missing_padding)
            
            decoded_bytes = base64.b64decode(value, validate=True)
            # Try to decode as UTF-8
            decoded = decoded_bytes.decode('utf-8', errors='ignore').strip()
            
            # Ensure it's actually different and valid
            if len(decoded) == 0 or decoded == value:
                return None
                
            return decoded
        except Exception:
            return None
    
    def _decode_base64url(self, value: str) -> Optional[str]:
        """Decode URL-safe Base64."""
        try:
            # Must look like base64url (alphanumeric + - _)
            if not re.match(r'^[A-Za-z0-9_-]+$', value):
                return None
            
            # Need minimum length
            if len(value) < 4:
                return None
            
            # If it's all hex characters, it's probably a hash, not base64url
            if all(c in '0123456789abcdefABCDEF' for c in value):
                # Check if it's a known hash length
                if len(value) in [32, 40, 56, 64, 96, 128]:
                    return None
            
            # Add padding if missing
            missing_padding = len(value) % 4
            if missing_padding:
                value += '=' * (4 - missing_padding)
            
            decoded_bytes = base64.urlsafe_b64decode(value)
            decoded = decoded_bytes.decode('utf-8', errors='ignore').strip()
            
            # Ensure it's actually different and valid
            if len(decoded) == 0 or decoded == value:
                return None
                
            return decoded
        except Exception:
            return None
    
    def _decode_hex(self, value: str) -> Optional[str]:
        """Decode hex encoding."""
        try:
            # Must be even length hex string
            if not re.match(r'^[0-9a-fA-F]+$', value) or len(value) % 2 != 0:
                return None
            
            # Don't decode if it looks like a hash (known hash lengths)
            if len(value) in [32, 40, 56, 64, 96, 128]:
                return None
            
            decoded_bytes = bytes.fromhex(value)
            decoded = decoded_bytes.decode('utf-8', errors='ignore').strip()
            
            # Ensure it's actually different and looks like text
            if len(decoded) == 0 or not any(c.isprintable() and c.isalnum() for c in decoded):
                return None
                
            return decoded
        except Exception:
            return None
    
    def _decode_url(self, value: str) -> Optional[str]:
        """Decode URL encoding."""
        try:
            # Must contain % encoding
            if '%' not in value:
                return None
            
            decoded = urllib.parse.unquote(value)
            return decoded if decoded != value else None
        except Exception:
            return None
    
    def _decode_quoted_printable(self, value: str) -> Optional[str]:
        """Decode quoted-printable encoding."""
        try:
            # Must contain = encoding
            if '=' not in value or not re.search(r'=[0-9A-Fa-f]{2}', value):
                return None
            
            import quopri
            decoded_bytes = quopri.decodestring(value.encode())
            return decoded_bytes.decode('utf-8', errors='ignore').strip()
        except Exception:
            return None
    
    def _decode_unicode_escape(self, value: str) -> Optional[str]:
        """Decode Unicode escape sequences."""
        try:
            # Must contain \u or \x escapes
            if not re.search(r'\\[ux]', value):
                return None
            
            decoded = value.encode().decode('unicode_escape')
            return decoded if decoded != value else None
        except Exception:
            return None
    
    def _is_valid_decoded(self, value: str) -> bool:
        """Check if decoded value looks valid."""
        # Must be printable ASCII or common hash characters
        if not value or len(value) < 4:
            return False
        
        # Should contain mostly printable characters
        printable_ratio = sum(c.isprintable() for c in value) / len(value)
        if printable_ratio < 0.8:
            return False
        
        # Check if it looks like a hash (hex string)
        if all(c in '0123456789abcdefABCDEF' for c in value):
            # If it's a valid hex hash length, it's good
            if len(value) in [32, 40, 56, 64, 96, 128]:
                return True
        
        # Check if it has reasonable entropy for text
        unique_chars = len(set(value))
        if unique_chars < 5:
            return False
        
        return True
    
    def _estimate_confidence(self, original: str, decoded: str, encoding: str) -> float:
        """Estimate confidence in decoding step."""
        # Base confidence by encoding type
        base_confidence = {
            'base64': 0.95,
            'base64url': 0.95,
            'hex': 0.90,
            'url': 0.85,
            'quoted_printable': 0.80,
            'unicode_escape': 0.75,
        }.get(encoding, 0.70)
        
        # Adjust based on length change
        length_ratio = len(decoded) / len(original) if len(original) > 0 else 0
        if length_ratio < 0.3 or length_ratio > 3.0:
            base_confidence *= 0.8
        
        # Adjust based on character validity
        printable_ratio = sum(c.isprintable() for c in decoded) / len(decoded)
        base_confidence *= printable_ratio
        
        return min(base_confidence, 1.0)


def decode_recursive(value: str, max_depth: int = 5) -> DecodingResult:
    """
    Convenience function for recursive decoding.
    
    Args:
        value: Input string to decode
        max_depth: Maximum recursion depth
        
    Returns:
        DecodingResult with chain and final value
    """
    decoder = RecursiveDecoder(max_depth=max_depth)
    return decoder.decode(value)
