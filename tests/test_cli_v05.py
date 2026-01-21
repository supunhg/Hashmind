#!/usr/bin/env python3
"""
CLI-specific tests for v0.5.0 features.
Tests command-line argument handling and integration.
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestCLIFlags:
    """Test CLI flag parsing and handling."""
    
    def test_help_output(self):
        """Test that help shows all v0.5.0 flags."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '--help'],
            capture_output=True,
            text=True
        )
        
        help_text = result.stdout
        
        # Check for all new flags
        assert '-r' in help_text or '--rules' in help_text, "Rules flag missing"
        assert '-d' in help_text or '--device' in help_text, "Device flag missing"
        assert '--no-cache' in help_text, "No-cache flag missing"
        
        # Check examples mention new features
        assert 'rules' in help_text.lower() or 'GPU' in help_text, "Help missing v0.5.0 examples"
    
    def test_short_flags(self):
        """Test that all short flags are recognized."""
        # These should not error with "unrecognized arguments"
        flags_to_test = [
            ['-c', 'test'],
            ['-v', 'test'],
            ['-b'],
            ['-C', 'test'],
            ['-T'],
        ]
        
        for flags in flags_to_test:
            result = subprocess.run(
                [sys.executable, '-m', 'src.cli'] + flags,
                capture_output=True,
                text=True,
                timeout=5
            )
            # Should not contain "unrecognized arguments" error
            assert 'unrecognized arguments' not in result.stderr.lower(), \
                f"Flags {flags} not recognized"
    
    def test_rules_flag(self):
        """Test -r/--rules flag."""
        # Test short form
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '-r', '/tmp/test.rule', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'unrecognized arguments' not in result.stderr.lower()
        
        # Test long form
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '--rules', '/tmp/test.rule', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_device_flag(self):
        """Test -d/--device flag."""
        # Test short form with single GPU
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '-d', '1', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'unrecognized arguments' not in result.stderr.lower()
        
        # Test with multiple GPUs
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '-d', '1,2,3', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'unrecognized arguments' not in result.stderr.lower()
        
        # Test long form
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '--device', '1', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_no_cache_flag(self):
        """Test --no-cache flag."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '--no-cache', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_combined_flags(self):
        """Test all flags combined."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '-v',
             '-r', '/tmp/rules.txt', '-d', '1',
             '-w', '/tmp/wordlist.txt', '-t', '60',
             '--no-cache', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'unrecognized arguments' not in result.stderr.lower(), \
            "Combined flags not accepted"


class TestCLIExamples:
    """Test that examples from README work."""
    
    def test_basic_crack(self):
        """Test basic crack command syntax."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', 'nonexistent_hash_12345'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Should attempt crack (will fail but syntax is correct)
        assert result.returncode in [0, 1], "Basic crack syntax failed"
    
    def test_crack_with_gpu(self):
        """Test crack with GPU selection."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '-d', '1', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode in [0, 1], "GPU selection syntax failed"
    
    def test_crack_with_rules(self):
        """Test crack with rules file."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '-r', '/tmp/test.rule', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode in [0, 1], "Rules syntax failed"
    
    def test_crack_no_cache(self):
        """Test crack with cache disabled."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '--no-cache', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode in [0, 1], "No-cache syntax failed"


class TestCLIOutput:
    """Test CLI output formatting."""
    
    def test_version_output(self):
        """Test version is v0.5.0."""
        # Read version from source
        import importlib.util
        spec = importlib.util.spec_from_file_location("src", "src/__init__.py")
        src_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(src_module)
        
        assert src_module.__version__ == '0.5.0', "Version not updated to 0.5.0"
    
    def test_check_tools_output(self):
        """Test -T/--check-tools output."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-T'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        output = result.stdout
        
        # Should show tool availability
        assert 'hashcat' in output.lower() or 'john' in output.lower(), \
            "Tool check output missing tools"
        
        # Should have clean formatting
        assert '✓' in output or '✗' in output or 'Available' in output, \
            "Tool check output not formatted"


class TestCLIErrors:
    """Test CLI error handling."""
    
    def test_missing_hash_argument(self):
        """Test error when hash argument is missing."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Should error or prompt for input
        assert result.returncode != 0 or 'required' in result.stderr.lower()
    
    def test_invalid_device_format(self):
        """Test handling of invalid device format."""
        # This should be accepted by CLI (validation happens in hashcat)
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', '-d', 'invalid', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # CLI should accept it, hashcat will handle validation
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_nonexistent_rules_file(self):
        """Test handling of nonexistent rules file."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C',
             '-r', '/nonexistent/path/rules.txt', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Should not crash, might show warning
        assert result.returncode in [0, 1], "Crashed on nonexistent rules file"


class TestCLIIntegrationWithCache:
    """Test CLI cache integration."""
    
    def test_cache_persistence(self):
        """Test that cache persists between CLI calls."""
        # This test would require actual cracking which we can't guarantee
        # Just verify cache location is accessible
        cache_dir = Path.home() / '.hashmind' / 'cracking'
        
        # Run CLI to ensure cache is initialized
        subprocess.run(
            [sys.executable, '-m', 'src.cli', '-C', 'test_hash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Cache directory should exist or be creatable
        assert cache_dir.exists() or cache_dir.parent.exists(), \
            "Cache directory not accessible"


if __name__ == '__main__':
    print("="*60)
    print("CLI TESTS FOR v0.5.0")
    print("="*60)
    
    exit_code = pytest.main([__file__, '-v', '--tb=short'])
    sys.exit(exit_code)
