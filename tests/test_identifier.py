"""Tests for the main identification API."""

import pytest
from src.identifier import identify, IdentificationResult


class TestIdentificationResult:
    """Test the IdentificationResult class."""
    
    def test_result_creation(self):
        matches = [
            {'algorithm': 'md5_hex', 'confidence': 0.9, 'reason': 'test', 'metadata': {}}
        ]
        result = IdentificationResult(matches, {'length': 32})
        
        assert result.top_match() == 'md5_hex'
        assert result.metadata['length'] == 32
    
    def test_empty_result(self):
        result = IdentificationResult([], {})
        assert result.top_match() is None
    
    def test_to_dict(self):
        matches = [
            {'algorithm': 'sha256_hex', 'confidence': 0.85, 'reason': 'test', 'metadata': {}}
        ]
        result = IdentificationResult(matches, {'length': 64})
        
        d = result.to_dict()
        assert d['top_match'] == 'sha256_hex'
        assert len(d['matches']) == 1
    
    def test_string_representation(self):
        matches = [
            {'algorithm': 'md5_hex', 'confidence': 0.9, 'reason': 'Length matches', 'metadata': {}},
            {'algorithm': 'md4_hex', 'confidence': 0.8, 'reason': 'Length matches', 'metadata': {}}
        ]
        result = IdentificationResult(matches, {})
        
        str_repr = str(result)
        assert 'md5_hex' in str_repr
        assert '90.00%' in str_repr


class TestIdentifyFunction:
    """Test the main identify() function."""
    
    def test_md5_identification(self):
        result = identify("5d41402abc4b2a76b9719d911017c592")
        
        assert result.top_match() == 'md5_hex'
        assert len(result.matches) > 0
        assert result.matches[0]['confidence'] > 0
    
    def test_sha256_identification(self):
        result = identify("2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae")
        
        top = result.top_match()
        assert top in ['sha256_hex', 'sha3_256_hex']
    
    def test_bcrypt_identification(self):
        bcrypt_hash = "$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy"
        result = identify(bcrypt_hash)
        
        assert result.top_match() == 'bcrypt'
        assert result.matches[0]['confidence'] > 0.8
    
    def test_jwt_identification(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = identify(jwt)
        
        assert result.top_match() == 'jwt'
    
    def test_uuid_identification(self):
        result = identify("550e8400-e29b-41d4-a716-446655440000")
        
        assert result.top_match() == 'uuid'
    
    def test_base64_identification(self):
        result = identify("SGVsbG8gV29ybGQ=")
        
        top = result.top_match()
        assert 'base64' in top
    
    def test_metadata_in_result(self):
        result = identify("5d41402abc4b2a76b9719d911017c592")
        
        assert 'length' in result.metadata
        assert 'entropy' in result.metadata
        assert 'charset' in result.metadata
        
        assert result.metadata['length'] == 32
    
    def test_multiple_matches(self):
        # MD5/MD4/NTLM all have same hex length
        result = identify("5d41402abc4b2a76b9719d911017c592")
        
        # Should have multiple potential matches
        assert len(result.matches) >= 1
        
        # All should have confidence scores
        for match in result.matches:
            assert 0 <= match['confidence'] <= 1
            assert 'reason' in match
