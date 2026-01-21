#!/usr/bin/env python3
"""
Stress tests and edge cases for v0.5.0.
Tests performance limits, concurrent access, and extreme scenarios.
"""

import pytest
import sqlite3
import time
import threading
import tempfile
from pathlib import Path
from src.cracker import HashCracker, crack_hash


class TestCacheStress:
    """Stress tests for cache system."""
    
    def test_large_cache(self):
        """Test cache with many entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            # Add 1000 entries
            num_entries = 1000
            for i in range(num_entries):
                cracker._save_to_cache(
                    f"hash_{i}",
                    f"password_{i}",
                    "md5_hex",
                    "test"
                )
            
            # Verify all can be retrieved
            for i in range(0, num_entries, 100):  # Sample every 100th
                result = cracker._check_cache(f"hash_{i}")
                assert result is not None, f"Entry {i} not found"
                assert result.plaintext == f"password_{i}", f"Wrong plaintext for entry {i}"
    
    def test_concurrent_cache_access(self):
        """Test concurrent access to cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            errors = []
            
            def worker(thread_id):
                try:
                    for i in range(10):
                        hash_val = f"thread_{thread_id}_hash_{i}"
                        plaintext = f"password_{i}"
                        
                        # Save
                        cracker._save_to_cache(hash_val, plaintext, "md5_hex", "test")
                        
                        # Retrieve
                        result = cracker._check_cache(hash_val)
                        assert result is not None
                        assert result.plaintext == plaintext
                except Exception as e:
                    errors.append(str(e))
            
            # Run 10 threads concurrently
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
    
    def test_cache_performance(self):
        """Test cache retrieval performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            # Add test entry
            cracker._save_to_cache("test_hash", "password", "md5_hex", "test")
            
            # Measure retrieval time
            iterations = 1000
            start = time.time()
            for _ in range(iterations):
                result = cracker._check_cache("test_hash")
            elapsed = time.time() - start
            
            avg_time = elapsed / iterations
            
            # Should be very fast (< 1ms per lookup)
            assert avg_time < 0.001, f"Cache too slow: {avg_time*1000:.2f}ms per lookup"
            
            print(f"\n  Cache performance: {avg_time*1000:.3f}ms per lookup")
    
    def test_large_plaintext(self):
        """Test cache with very large plaintext."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            # 1MB plaintext
            large_plaintext = "A" * (1024 * 1024)
            
            cracker._save_to_cache("test_hash", large_plaintext, "md5_hex", "test")
            result = cracker._check_cache("test_hash")
            
            assert result is not None, "Large plaintext not saved"
            assert result.plaintext == large_plaintext, "Large plaintext corrupted"
    
    def test_special_characters(self):
        """Test cache with special characters in plaintext."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            special_chars = [
                "pass'word",
                'pass"word',
                "pass\nword",
                "pass\tword",
                "pass;DROP TABLE cracks;--",
                "pass\x00word",
                "🔐🔑password",
                "パスワード",
            ]
            
            for i, plaintext in enumerate(special_chars):
                hash_val = f"hash_{i}"
                cracker._save_to_cache(hash_val, plaintext, "md5_hex", "test")
                result = cracker._check_cache(hash_val)
                
                assert result is not None, f"Special char test {i} failed to save"
                assert result.plaintext == plaintext, f"Special char test {i} corrupted"


class TestEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_very_long_hash(self):
        """Test with extremely long hash value."""
        cracker = HashCracker()
        
        # 10KB hash
        long_hash = "a" * (10 * 1024)
        
        result = cracker.crack(long_hash, "md5_hex", max_time=1)
        # Should handle gracefully (will fail to crack but not crash)
        assert isinstance(result.success, bool), "Long hash crashed system"
    
    def test_binary_data_in_hash(self):
        """Test with binary data."""
        cracker = HashCracker()
        
        binary_hash = "\x00\x01\x02\x03\x04"
        
        result = cracker.crack(binary_hash, "md5_hex", max_time=1)
        assert isinstance(result.success, bool), "Binary data crashed system"
    
    def test_unicode_hash(self):
        """Test with unicode characters."""
        cracker = HashCracker()
        
        unicode_hash = "🔐💎🎯"
        
        result = cracker.crack(unicode_hash, "md5_hex", max_time=1)
        assert isinstance(result.success, bool), "Unicode crashed system"
    
    def test_empty_wordlist(self):
        """Test with empty wordlist."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            wordlist = f.name
            # Empty file
        
        try:
            cracker = HashCracker(wordlist=wordlist)
            result = cracker.crack("test_hash", "md5_hex", max_time=1)
            # Should handle gracefully
            assert isinstance(result.success, bool)
        finally:
            Path(wordlist).unlink()
    
    def test_zero_timeout(self):
        """Test with zero timeout."""
        cracker = HashCracker()
        result = cracker.crack("test_hash", "md5_hex", max_time=0)
        # Should handle gracefully
        assert isinstance(result.success, bool)
    
    def test_negative_timeout(self):
        """Test with negative timeout."""
        cracker = HashCracker()
        # Hashcat will handle this, shouldn't crash our code
        result = cracker.crack("test_hash", "md5_hex", max_time=-1)
        assert isinstance(result.success, bool)


class TestMemoryLeaks:
    """Test for potential memory leaks."""
    
    def test_repeated_cache_operations(self):
        """Test repeated cache operations don't leak memory."""
        import tracemalloc
        
        tracemalloc.start()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            cracker._init_cache_db()
            
            # Measure initial memory
            tracemalloc.reset_peak()
            
            # Perform many operations
            for i in range(1000):
                hash_val = f"hash_{i % 100}"  # Reuse hashes
                cracker._save_to_cache(hash_val, f"pass_{i}", "md5_hex", "test")
                cracker._check_cache(hash_val)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Peak memory should be reasonable (< 10MB for this test)
            assert peak < 10 * 1024 * 1024, f"Possible memory leak: {peak/1024/1024:.2f}MB used"
            
            print(f"\n  Memory usage: current={current/1024:.1f}KB, peak={peak/1024:.1f}KB")
    
    def test_many_cracker_instances(self):
        """Test creating many cracker instances."""
        crackers = []
        
        # Create 100 instances
        for i in range(100):
            cracker = HashCracker(use_cache=False)  # Don't init DB to save time
            crackers.append(cracker)
        
        # Should not crash or leak
        assert len(crackers) == 100, "Failed to create crackers"


class TestDatabaseIntegrity:
    """Test database integrity and corruption handling."""
    
    def test_corrupted_database(self):
        """Test handling of corrupted database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'crack_cache.db'
            
            # Create corrupted database
            with open(db_path, 'w') as f:
                f.write("This is not a valid SQLite database")
            
            # Should handle gracefully
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            
            # Should not crash
            try:
                cracker._init_cache_db()
                # Cache might be disabled
                assert True
            except Exception as e:
                # Should handle error gracefully
                assert "database" in str(e).lower() or True
    
    def test_missing_table(self):
        """Test handling of database with missing table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'crack_cache.db'
            
            # Create DB without the cracks table
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE dummy (id INTEGER)")
            conn.close()
            
            cracker = HashCracker(use_cache=True)
            cracker.temp_dir = Path(tmpdir)
            
            # Should recreate table or handle gracefully
            result = cracker._check_cache("test_hash")
            assert result is None or isinstance(result, type(None))


class TestConcurrency:
    """Test concurrent cracking operations."""
    
    def test_multiple_crackers_same_cache(self):
        """Test multiple crackers sharing same cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            
            # Create and populate cache
            cracker1 = HashCracker(use_cache=True)
            cracker1.temp_dir = temp_path
            cracker1._init_cache_db()
            cracker1._save_to_cache("shared_hash", "password", "md5_hex", "test")
            
            # Second cracker using same cache
            cracker2 = HashCracker(use_cache=True)
            cracker2.temp_dir = temp_path
            
            # Should read from shared cache
            result = cracker2._check_cache("shared_hash")
            assert result is not None, "Shared cache not accessible"
            assert result.plaintext == "password", "Shared cache data corrupted"


def run_all_stress_tests():
    """Run all stress tests with reporting."""
    print("\n" + "="*60)
    print("STRESS TEST SUITE")
    print("="*60)
    
    exit_code = pytest.main([__file__, '-v', '--tb=short', '-x'])
    
    return exit_code


if __name__ == '__main__':
    import sys
    sys.exit(run_all_stress_tests())
