"""Tests for the CLI interface."""

import pytest
from src.cli import main
from io import StringIO
import sys


class TestCLI:
    """Test command-line interface."""
    
    def test_basic_identification(self, capsys):
        """Test basic hash identification."""
        exit_code = main(["5d41402abc4b2a76b9719d911017c592"])
        
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "md5_hex" in captured.out
    
    def test_confidence_mode(self, capsys):
        """Test --confidence flag."""
        exit_code = main(["--confidence", "5d41402abc4b2a76b9719d911017c592"])
        
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "%" in captured.out
        assert "md5_hex" in captured.out
    
    def test_verbose_mode(self, capsys):
        """Test --verbose flag."""
        exit_code = main(["--verbose", "5d41402abc4b2a76b9719d911017c592"])
        
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Metadata" in captured.out
        assert "Entropy" in captured.out
    
    def test_unknown_input(self, capsys):
        """Test with unknown/ambiguous input."""
        exit_code = main(["randomstring123"])
        
        assert exit_code == 0
        # Should still return something (even if "unknown")
    
    def test_bcrypt_hash(self, capsys):
        """Test bcrypt hash identification."""
        bcrypt = "$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy"
        exit_code = main([bcrypt])
        
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "bcrypt" in captured.out
    
    def test_no_arguments_no_stdin(self, capsys, monkeypatch):
        """Test running with no arguments and no stdin."""
        # Simulate terminal (isatty returns True)
        monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
        
        exit_code = main([])
        assert exit_code == 1  # Should show help and exit
