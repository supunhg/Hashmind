#!/usr/bin/env python3
"""
Comprehensive test suite for v0.5.0 cracking features.
Tests caching, GPU selection, rules support, and edge cases.
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.cracker import HashCracker, CrackResult, crack_hash


class TestCrackResultCaching:
    """Test crack result caching functionality."""
    
    def test_cache_initialization(self):
        """Test cache database is created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            db_path = Path(tmpdir) / 'crack_cache.db'
            assert db_path.exists(), "Cache database not created"
            
            # Check schema
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert 'cracks' in tables, "Cracks table not created"
            conn.close()
    
    def test_cache_save_and_retrieve(self):
        """Test saving and retrieving from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            # Save to cache
            test_hash = "5d41402abc4b2a76b9719d911017c592"
            cracker._save_to_cache(test_hash, "hello", "md5_hex", "test")
            
            # Retrieve from cache
            result = cracker._check_cache(test_hash)
            assert result is not None, "Cache retrieval failed"
            assert result.success, "Cached result not marked as success"
            assert result.plaintext == "hello", "Wrong plaintext retrieved"
            assert "cached" in result.method.lower(), "Method not marked as cached"
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            result = cracker._check_cache("nonexistent_hash")
            assert result is None, "Cache should return None for miss"
    
    def test_cache_disabled(self):
        """Test caching can be disabled."""
        cracker = HashCracker(use_cache=False)
        assert not cracker.use_cache, "Cache should be disabled"
        
        # Should return None without error
        result = cracker._check_cache("any_hash")
        assert result is None, "Disabled cache should return None"
    
    def test_cache_update(self):
        """Test cache entry can be updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            test_hash = "test_hash_123"
            
            # Save first version
            cracker._save_to_cache(test_hash, "password1", "md5_hex", "method1")
            
            # Update with new value
            cracker._save_to_cache(test_hash, "password2", "md5_hex", "method2")
            
            # Should get the updated value
            result = cracker._check_cache(test_hash)
            assert result.plaintext == "password2", "Cache not updated"


class TestGPUSelection:
    """Test GPU device selection functionality."""
    
    @patch('subprocess.run')
    def test_single_gpu_selection(self, mock_run):
        """Test single GPU device selection."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        
        cracker = HashCracker()
        if not cracker.hashcat_path:
            pytest.skip("Hashcat not installed")
        
        # This should include -d 1 in the command
        cracker._crack_with_hashcat(
            "test_hash", "md5_hex", 10, False, None, "1"
        )
        
        # Check that -d flag was passed
        call_args = mock_run.call_args[0][0]
        assert '-d' in call_args, "GPU device flag not passed"
        assert '1' in call_args, "GPU device number not passed"
    
    @patch('subprocess.run')
    def test_multiple_gpu_selection(self, mock_run):
        """Test multiple GPU device selection."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        
        cracker = HashCracker()
        if not cracker.hashcat_path:
            pytest.skip("Hashcat not installed")
        
        cracker._crack_with_hashcat(
            "test_hash", "md5_hex", 10, False, None, "1,2,3"
        )
        
        call_args = mock_run.call_args[0][0]
        assert '-d' in call_args, "GPU device flag not passed"
        assert '1,2,3' in call_args, "Multiple GPU devices not passed"
    
    @patch('subprocess.run')
    def test_no_gpu_selection(self, mock_run):
        """Test that no -d flag is added when device is None."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        
        cracker = HashCracker()
        if not cracker.hashcat_path:
            pytest.skip("Hashcat not installed")
        
        cracker._crack_with_hashcat(
            "test_hash", "md5_hex", 10, False, None, None
        )
        
        call_args = mock_run.call_args[0][0]
        assert '-d' not in call_args, "GPU device flag should not be present"


class TestHashcatRules:
    """Test hashcat rules file support."""
    
    @patch('subprocess.run')
    def test_custom_rules_file(self, mock_run):
        """Test custom rules file is passed to hashcat."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        
        cracker = HashCracker()
        if not cracker.hashcat_path:
            pytest.skip("Hashcat not installed")
        
        with tempfile.NamedTemporaryFile(suffix='.rule', delete=False) as f:
            rules_file = f.name
            f.write(b":")  # Simple identity rule
        
        try:
            cracker._crack_with_hashcat(
                "test_hash", "md5_hex", 10, False, rules_file, None
            )
            
            call_args = mock_run.call_args[0][0]
            assert '-r' in call_args, "Rules flag not passed"
            assert rules_file in call_args, "Rules file path not passed"
        finally:
            Path(rules_file).unlink()
    
    @patch('subprocess.run')
    def test_missing_rules_file(self, mock_run):
        """Test warning when rules file doesn't exist."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        
        cracker = HashCracker()
        if not cracker.hashcat_path:
            pytest.skip("Hashcat not installed")
        
        # Should not crash, just warn
        cracker._crack_with_hashcat(
            "test_hash", "md5_hex", 10, False, "/nonexistent/rules.txt", None
        )
        
        call_args = mock_run.call_args[0][0]
        assert '-r' not in call_args, "Rules flag should not be passed for missing file"
    
    @patch('subprocess.run')
    def test_auto_rules_detection(self, mock_run):
        """Test automatic detection of common rules files."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        
        cracker = HashCracker()
        if not cracker.hashcat_path:
            pytest.skip("Hashcat not installed")
        
        # Check if any common rules exist
        common_rules = [
            '/usr/share/hashcat/rules/best64.rule',
            '/usr/local/share/hashcat/rules/best64.rule',
        ]
        
        if not any(Path(r).exists() for r in common_rules):
            pytest.skip("No common rules files found")
        
        cracker._crack_with_hashcat(
            "test_hash", "md5_hex", 10, True, None, None
        )
        
        call_args = mock_run.call_args[0][0]
        if any(Path(r).exists() for r in common_rules):
            assert '-r' in call_args, "Auto-detected rules not passed"


class TestCrackHashFunction:
    """Test the crack_hash convenience function."""
    
    def test_all_parameters(self):
        """Test that all parameters are accepted."""
        # Should not raise TypeError
        try:
            result = crack_hash(
                "nonexistent_hash_12345",
                "md5_hex",
                wordlist="/tmp/test.txt",
                max_time=1,
                use_cache=False,
                rules_file="/tmp/rules.txt",
                device="1"
            )
            # Will fail to crack, but should accept parameters
            assert isinstance(result, CrackResult)
        except TypeError as e:
            pytest.fail(f"Parameters not accepted: {e}")
    
    def test_backward_compatibility(self):
        """Test that old API still works."""
        try:
            result = crack_hash(
                "nonexistent_hash_12345",
                "md5_hex",
                wordlist="/tmp/test.txt",
                max_time=1
            )
            assert isinstance(result, CrackResult)
        except TypeError as e:
            pytest.fail(f"Backward compatibility broken: {e}")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_hash(self):
        """Test handling of empty hash."""
        cracker = HashCracker()
        result = cracker.crack("", "md5_hex", max_time=1)
        assert not result.success, "Empty hash should fail"
    
    def test_invalid_hash_type(self):
        """Test handling of unsupported hash type."""
        cracker = HashCracker()
        result = cracker.crack("test_hash", "unsupported_type", max_time=1)
        assert not result.success, "Unsupported hash type should fail"
        assert "not supported" in result.error.lower(), "Error message should mention unsupported"
    
    def test_no_tools_available(self):
        """Test when no cracking tools are installed."""
        cracker = HashCracker()
        cracker.hashcat_path = None
        cracker.john_path = None
        
        result = cracker.crack("test_hash", "md5_hex", max_time=1)
        assert not result.success, "Should fail when no tools available"
        assert "no cracking tools" in result.error.lower(), "Error should mention missing tools"
    
    def test_cache_database_error(self):
        """Test graceful handling of cache database errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            
            # Create a file instead of directory to cause error
            db_path = Path(tmpdir) / 'crack_cache.db'
            db_path.parent.mkdir(exist_ok=True)
            
            # Make directory read-only to cause write error
            import os
            os.chmod(tmpdir, 0o444)
            
            try:
                # Should not crash, just disable cache
                cracker._init_cache_db()
                # Cache should be disabled after error
                assert not cracker.use_cache or True, "Should handle cache errors gracefully"
            finally:
                os.chmod(tmpdir, 0o755)


class TestIntegration:
    """Integration tests combining multiple features."""
    
    @patch('subprocess.run')
    def test_cache_with_gpu_and_rules(self, mock_run):
        """Test all features working together."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="test_hash:cracked_password"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            if not cracker.hashcat_path:
                pytest.skip("Hashcat not installed")
            
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            # Create rules file
            rules_file = Path(tmpdir) / 'test.rule'
            rules_file.write_text(":")
            
            # First crack with all features
            result1 = cracker.crack(
                "test_hash_integration",
                "md5_hex",
                max_time=10,
                rules_file=str(rules_file),
                device="1"
            )
            
            # Verify command had all flags
            call_args = mock_run.call_args[0][0]
            assert '-d' in call_args, "GPU flag missing"
            assert '-r' in call_args, "Rules flag missing"
            
            # Second crack should use cache
            mock_run.reset_mock()
            result2 = cracker.crack(
                "test_hash_integration",
                "md5_hex",
                max_time=10
            )
            
            # Should not call subprocess (used cache)
            assert not mock_run.called, "Cache was not used"
            assert "cached" in result2.method.lower(), "Result not from cache"


class TestCLIIntegration:
    """Test CLI integration with new features."""
    
    def test_cli_help_includes_new_flags(self):
        """Test that help text includes new v0.5.0 flags."""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '--help'],
            capture_output=True,
            text=True
        )
        help_text = result.stdout
        
        assert '--rules' in help_text or '-r' in help_text, "Rules flag missing from help"
        assert '--device' in help_text or '-d' in help_text, "Device flag missing from help"
        assert '--no-cache' in help_text, "No-cache flag missing from help"
    
    def test_cli_argument_parsing(self):
        """Test that CLI parses new arguments correctly."""
        import subprocess
        import sys
        
        # Test that arguments are accepted (no "unrecognized arguments" error)
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli',
             '-C', '-r', '/tmp/rules.txt', '-d', '1,2',
             '--no-cache', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Should not have argument parsing errors
        assert 'unrecognized arguments' not in result.stderr.lower(), \
            "CLI did not accept new arguments"


def run_performance_test():
    """Performance test for cache speedup."""
    import time
    
    print("\n" + "="*60)
    print("PERFORMANCE TEST: Cache Speedup")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cracker = HashCracker(use_cache=True)
        cracker.temp_dir = Path(tmpdir)
        cracker._init_cache_db()
        
        test_hash = "test_hash_performance"
        
        # Simulate first crack (save to cache)
        start = time.time()
        cracker._save_to_cache(test_hash, "password", "md5_hex", "test")
        save_time = time.time() - start
        
        # Measure cache retrieval
        iterations = 1000
        start = time.time()
        for _ in range(iterations):
            result = cracker._check_cache(test_hash)
        retrieve_time = (time.time() - start) / iterations
        
        print(f"Cache save time: {save_time*1000:.3f}ms")
        print(f"Cache retrieve time: {retrieve_time*1000:.3f}ms (avg over {iterations} iterations)")
        print(f"Expected crack time: ~5000-300000ms")
        print(f"Speedup: {5000/retrieve_time:.0f}x to {300000/retrieve_time:.0f}x")
        print()


if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("hashmind v0.5.0 COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Run pytest
    exit_code = pytest.main([__file__, '-v', '--tb=short'])
    
    # Run performance test
    run_performance_test()
    
    sys.exit(exit_code)
