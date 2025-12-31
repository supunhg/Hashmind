"""Tests for the core detection engine."""

import pytest
from src.core.detector import DetectorPipeline
from src.core.matchers import LengthMatcher, PrefixSuffixDetector, RegexEngine
from src.core.analyzers import CharacterSetAnalyzer, EntropyCalculator


class TestLengthMatcher:
    """Test length-based matching."""
    
    def test_md5_hex_length(self):
        matcher = LengthMatcher()
        matches = matcher.match("5d41402abc4b2a76b9719d911017c592")
        
        algorithms = [m.algorithm for m in matches]
        assert 'md5_hex' in algorithms
        assert len(matches) > 0
    
    def test_sha256_hex_length(self):
        matcher = LengthMatcher()
        matches = matcher.match("2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae")
        
        algorithms = [m.algorithm for m in matches]
        assert 'sha256_hex' in algorithms
    
    def test_bcrypt_length_range(self):
        matcher = LengthMatcher()
        # Bcrypt hashes are 60 characters
        bcrypt_hash = "$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy"
        matches = matcher.match(bcrypt_hash)
        
        algorithms = [m.algorithm for m in matches]
        assert 'bcrypt' in algorithms


class TestPrefixSuffixDetector:
    """Test prefix/suffix detection."""
    
    def test_bcrypt_prefix(self):
        detector = PrefixSuffixDetector()
        matches = detector.match("$2a$10$somesaltandhashhereblahblah")
        
        assert len(matches) > 0
        assert matches[0].algorithm == 'bcrypt'
        assert matches[0].confidence > 0.8
    
    def test_sha512crypt_prefix(self):
        detector = PrefixSuffixDetector()
        matches = detector.match("$6$rounds=5000$somesalt$hash")
        
        algorithms = [m.algorithm for m in matches]
        assert 'sha512crypt' in algorithms
    
    def test_jwt_pattern(self):
        detector = PrefixSuffixDetector()
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        matches = detector.match(jwt)
        
        assert len(matches) > 0
        assert matches[0].algorithm == 'jwt'
        assert matches[0].confidence > 0.9


class TestRegexEngine:
    """Test regex pattern matching."""
    
    def test_uuid_pattern(self):
        engine = RegexEngine()
        matches = engine.match("550e8400-e29b-41d4-a716-446655440000")
        
        algorithms = [m.algorithm for m in matches]
        assert 'uuid' in algorithms
    
    def test_hex_pattern(self):
        engine = RegexEngine()
        matches = engine.match("deadbeef")
        
        algorithms = [m.algorithm for m in matches]
        assert 'hex' in algorithms
    
    def test_base64_pattern(self):
        engine = RegexEngine()
        matches = engine.match("SGVsbG8gV29ybGQ=")
        
        algorithms = [m.algorithm for m in matches]
        assert 'base64' in algorithms
    
    def test_ethereum_address(self):
        engine = RegexEngine()
        matches = engine.match("0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb")
        
        algorithms = [m.algorithm for m in matches]
        assert 'ethereum_address' in algorithms


class TestCharacterSetAnalyzer:
    """Test character set analysis."""
    
    def test_hex_detection(self):
        analyzer = CharacterSetAnalyzer()
        result = analyzer.analyze("deadbeef1234567890abcdef")
        
        assert result['is_hex'] is True
        assert result['has_lowercase'] is True
    
    def test_base64_detection(self):
        analyzer = CharacterSetAnalyzer()
        result = analyzer.analyze("SGVsbG8gV29ybGQ=")
        
        assert result['is_base64'] is True
        assert result['has_uppercase'] is True
        assert result['has_lowercase'] is True
    
    def test_mixed_charset(self):
        analyzer = CharacterSetAnalyzer()
        result = analyzer.analyze("Hello123!@#")
        
        assert result['has_special'] is True
        assert result['has_digits'] is True
        assert result['has_uppercase'] is True
        assert result['has_lowercase'] is True


class TestEntropyCalculator:
    """Test entropy calculations."""
    
    def test_high_entropy(self):
        calc = EntropyCalculator()
        # Random-looking hash should have high entropy
        entropy = calc.shannon_entropy("5d41402abc4b2a76b9719d911017c592")
        
        assert entropy > 3.5  # Hex strings typically have ~4 bits per char
    
    def test_low_entropy(self):
        calc = EntropyCalculator()
        # Repetitive string has low entropy
        entropy = calc.shannon_entropy("aaaaaaaaaa")
        
        assert entropy < 1.0
    
    def test_compression_ratio(self):
        calc = EntropyCalculator()
        
        # Random data compresses poorly
        ratio1 = calc.compression_ratio("5d41402abc4b2a76b9719d911017c592")
        
        # Repetitive data compresses well
        ratio2 = calc.compression_ratio("aaaaaaaaaaaaaaaa")
        
        assert ratio1 > ratio2


class TestDetectorPipeline:
    """Test the full detection pipeline."""
    
    def test_md5_identification(self):
        pipeline = DetectorPipeline()
        result = pipeline.analyze("5d41402abc4b2a76b9719d911017c592")
        
        assert result['success'] is True
        assert len(result['matches']) > 0
        
        # MD5 should be in top matches
        algorithms = [m.algorithm for m in result['matches']]
        assert 'md5_hex' in algorithms
    
    def test_bcrypt_identification(self):
        pipeline = DetectorPipeline()
        bcrypt_hash = "$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy"
        result = pipeline.analyze(bcrypt_hash)
        
        assert result['success'] is True
        assert result['matches'][0].algorithm == 'bcrypt'
        assert result['matches'][0].confidence > 0.8
    
    def test_jwt_identification(self):
        pipeline = DetectorPipeline()
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = pipeline.analyze(jwt)
        
        assert result['success'] is True
        assert result['matches'][0].algorithm == 'jwt'
    
    def test_quick_identify(self):
        pipeline = DetectorPipeline()
        
        assert pipeline.quick_identify("5d41402abc4b2a76b9719d911017c592") == "md5_hex"
        assert pipeline.quick_identify("unknown_format_12345") in ["unknown", "hex"]
    
    def test_empty_input_handling(self):
        pipeline = DetectorPipeline()
        result = pipeline.analyze("")
        
        assert result['success'] is False
        assert 'error' in result['metadata']
    
    def test_metadata_collection(self):
        pipeline = DetectorPipeline()
        result = pipeline.analyze("5d41402abc4b2a76b9719d911017c592")
        
        assert 'length' in result['metadata']
        assert 'entropy' in result['metadata']
        assert 'charset' in result['metadata']
        
        assert result['metadata']['length'] == 32
        assert 'shannon' in result['metadata']['entropy']
