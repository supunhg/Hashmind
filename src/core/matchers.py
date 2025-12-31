"""Fast heuristic matchers for hash and format detection."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Match:
    """Represents a detection match."""
    algorithm: str
    confidence: float
    reason: str
    metadata: Dict[str, Any]


class LengthMatcher:
    """Match hashes based on exact or range-based length."""
    
    # Exact length mappings for common hash algorithms
    EXACT_LENGTHS = {
        # Hex-encoded hashes
        8: ['crc32_hex', 'adler32_hex', 'crc32_int', 'adler32_int'],
        13: ['mysql_old'],
        16: ['md5_raw', 'md4_raw', 'md2_raw', 'oracle10g'],
        20: ['sha1_raw', 'ripemd160_raw'],
        22: ['md5_base64'],
        24: ['md5_base64_padded', 'tiger192_raw'],
        27: ['sha1_base64'],
        28: ['sha1_base64_padded', 'sha224_raw', 'sha3_224_raw'],
        32: ['md5_hex', 'md4_hex', 'ntlm_hex', 'md2_hex', 'ripemd128_hex', 'tiger128_hex'],
        34: ['oracle11g'],
        36: ['uuid_standard'],
        38: ['sha224_base64'],
        40: ['sha1_hex', 'ripemd160_hex', 'tiger160_hex', 'haval160_hex', 'dsa_hex'],
        43: ['sha256_base64'],
        44: ['sha256_base64_padded'],
        48: ['tiger192_hex', 'haval192_hex'],
        56: ['sha224_hex', 'sha3_224_hex', 'haval224_hex'],
        64: ['sha256_hex', 'sha3_256_hex', 'blake2s_hex', 'haval256_hex', 'ripemd256_hex', 'gost_hex', 'snefru256_hex', 'sha384_base64'],
        80: ['ripemd320_hex'],
        86: ['sha512_base64'],
        88: ['sha512_base64_padded'],
        96: ['sha384_hex', 'sha3_384_hex'],
        128: ['sha512_hex', 'sha3_512_hex', 'blake2b_hex', 'whirlpool_hex', 'sha512_224_hex', 'sha512_256_hex'],
    }
    
    # Common hash length ranges
    LENGTH_RANGES = {
        (60, 60): ['bcrypt'],  # $2a$10$...
        (13, 13): ['mysql_old'],
        (41, 41): ['mysql_new'],  # *HASH
    }
    
    def match(self, input_string: str) -> List[Match]:
        """
        Match based on string length.
        
        Args:
            input_string: Input string to analyze
            
        Returns:
            List of possible matches
        """
        matches = []
        length = len(input_string)
        
        # Pre-check character sets for better accuracy
        is_hex = all(c in '0123456789abcdefABCDEF' for c in input_string)
        is_base64_chars = all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in input_string)
        
        # Check exact length matches
        if length in self.EXACT_LENGTHS:
            for algo in self.EXACT_LENGTHS[length]:
                # Filter based on character set
                if '_hex' in algo and not is_hex:
                    continue
                if '_base64' in algo and not is_base64_chars:
                    continue
                
                # Boost confidence for hex hashes if chars match
                confidence = 0.7
                if '_hex' in algo and is_hex:
                    confidence = 0.75
                elif '_base64' in algo and is_base64_chars:
                    confidence = 0.72
                    
                matches.append(Match(
                    algorithm=algo,
                    confidence=confidence,
                    reason=f"Matches {algo} length ({length} chars)",
                    metadata={'length': length}
                ))
        
        # Check range-based matches
        for (min_len, max_len), algos in self.LENGTH_RANGES.items():
            if min_len <= length <= max_len:
                for algo in algos:
                    matches.append(Match(
                        algorithm=algo,
                        confidence=0.6,
                        reason=f"Length in {algo} range ({min_len}-{max_len})",
                        metadata={'length': length}
                    ))
        
        return matches


class PrefixSuffixDetector:
    """Detect based on known prefixes/suffixes (magic bytes)."""
    
    # Known prefixes (sorted by specificity - more specific first)
    PREFIXES = {
        # Bcrypt variants
        '$2a$': 'bcrypt',
        '$2b$': 'bcrypt',
        '$2y$': 'bcrypt',
        '$2x$': 'bcrypt',
        
        # Unix crypt variants
        '$6$': 'sha512crypt',
        '$5$': 'sha256crypt',
        '$1$': 'md5crypt',
        '$apr1$': 'apr_md5',
        
        # Argon2 variants
        '$argon2i$': 'argon2i',
        '$argon2d$': 'argon2d',
        '$argon2id$': 'argon2id',
        '$argon2': 'argon2',
        
        # PBKDF2 variants
        '$pbkdf2-sha512$': 'pbkdf2_sha512',
        '$pbkdf2-sha256$': 'pbkdf2_sha256',
        '$pbkdf2$': 'pbkdf2',
        
        # LDAP formats
        '{SHA}': 'ldap_sha1',
        '{SSHA}': 'ldap_ssha',
        '{MD5}': 'ldap_md5',
        '{SMD5}': 'ldap_smd5',
        '{CRYPT}': 'ldap_crypt',
        '{CLEAR}': 'ldap_clear',
        
        # Django formats
        'pbkdf2_sha256$': 'django_pbkdf2_sha256',
        'pbkdf2_sha1$': 'django_pbkdf2_sha1',
        'bcrypt_sha256$': 'django_bcrypt_sha256',
        'bcrypt$': 'django_bcrypt',
        'sha1$': 'django_sha1',
        'md5$': 'django_md5',
        'argon2$': 'django_argon2',
        
        # Other frameworks
        'scrypt:': 'scrypt',
        '$S$': 'drupal7',
        '$P$': 'phpass',
        '$H$': 'phpass',
        
        # Oracle
        'S:': 'oracle11g',
        
        # PostgreSQL
        'md5': 'postgres_md5',
        'SCRAM-SHA-256$': 'postgres_scram_sha256',
    }
    
    # JWT pattern (header.payload.signature)
    JWT_PATTERN = re.compile(r'^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$')
    
    def match(self, input_string: str) -> List[Match]:
        """
        Match based on prefixes/suffixes.
        
        Args:
            input_string: Input string to analyze
            
        Returns:
            List of possible matches
        """
        matches = []
        
        # Check known prefixes
        for prefix, algo in self.PREFIXES.items():
            if input_string.startswith(prefix):
                matches.append(Match(
                    algorithm=algo,
                    confidence=0.9,  # Prefixes are high confidence
                    reason=f"Starts with {algo} prefix: {prefix}",
                    metadata={'prefix': prefix}
                ))
        
        # Special case: MySQL new format (starts with * and 40 hex chars)
        if input_string.startswith('*') and len(input_string) == 41:
            if all(c in '0123456789ABCDEFabcdef' for c in input_string[1:]):
                matches.append(Match(
                    algorithm='mysql_new',
                    confidence=0.85,
                    reason="Matches MySQL new format (*{40 hex chars})",
                    metadata={'prefix': '*'}
                ))
        
        # Check JWT pattern
        if self.JWT_PATTERN.match(input_string):
            parts = input_string.split('.')
            matches.append(Match(
                algorithm='jwt',
                confidence=0.95,
                reason="Matches JWT structure (header.payload.signature)",
                metadata={'parts': len(parts)}
            ))
        
        return matches


class RegexEngine:
    """Pattern matching for complex formats."""
    
    # Regex patterns for various formats
    PATTERNS = {
        # Identifiers
        'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE),
        'uuid_no_dash': re.compile(r'^[0-9a-f]{32}$', re.IGNORECASE),
        'guid': re.compile(r'^\{[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}$', re.IGNORECASE),
        
        # Cryptocurrency
        'bitcoin_address': re.compile(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$'),
        'bitcoin_bech32': re.compile(r'^bc1[a-z0-9]{39,87}$'),
        'ethereum_address': re.compile(r'^0x[a-fA-F0-9]{40}$'),
        'litecoin_address': re.compile(r'^[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}$'),
        'ripple_address': re.compile(r'^r[1-9A-HJ-NP-Za-km-z]{25,34}$'),
        
        # Encodings
        'base64': re.compile(r'^[A-Za-z0-9+/]+=*$'),
        'base64url': re.compile(r'^[A-Za-z0-9_-]+=*$'),
        'base32': re.compile(r'^[A-Z2-7]+=*$'),
        'base58': re.compile(r'^[1-9A-HJ-NP-Za-km-z]+$'),
        'hex': re.compile(r'^[a-fA-F0-9]+$'),
        'url_encoded': re.compile(r'^[A-Za-z0-9\-_.~%]+$'),
        
        # Network
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'ipv4': re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
        'ipv6': re.compile(r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'),
        'ipv6_compressed': re.compile(r'^(([0-9a-fA-F]{1,4}:){1,7}:|:((:[0-9a-fA-F]{1,4}){1,7}|:))$'),
        'mac_address': re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'),
        'url': re.compile(r'^https?://[^\s]+$'),
        
        # File formats
        'md5sum_file': re.compile(r'^[a-f0-9]{32}\s+\*?[\w\-./]+$', re.IGNORECASE),
        'sha1sum_file': re.compile(r'^[a-f0-9]{40}\s+\*?[\w\-./]+$', re.IGNORECASE),
        'sha256sum_file': re.compile(r'^[a-f0-9]{64}\s+\*?[\w\-./]+$', re.IGNORECASE),
        
        # Other
        'windows_sid': re.compile(r'^S-1-[0-59]-\d{2}-\d{8,10}-\d{8,10}-\d{8,10}-[1-9]\d{3}$'),
        'jwt_header': re.compile(r'^eyJ[A-Za-z0-9_-]+$'),
    }
    
    def match(self, input_string: str) -> List[Match]:
        """
        Match against regex patterns.
        
        Args:
            input_string: Input string to analyze
            
        Returns:
            List of possible matches
        """
        matches = []
        
        for pattern_name, pattern in self.PATTERNS.items():
            if pattern.match(input_string):
                # Adjust confidence based on pattern specificity
                if pattern_name in ['uuid', 'bitcoin_address', 'ethereum_address', 'bitcoin_bech32']:
                    confidence = 0.85
                elif pattern_name in ['windows_sid', 'jwt_header', 'mac_address']:
                    confidence = 0.90
                elif pattern_name in ['hex', 'base64', 'base32']:
                    confidence = 0.4  # Very generic
                else:
                    confidence = 0.6
                    
                matches.append(Match(
                    algorithm=pattern_name,
                    confidence=confidence,
                    reason=f"Matches {pattern_name} pattern",
                    metadata={'pattern': pattern_name}
                ))
        
        return matches
