#!/usr/bin/env python3
"""
Quick test script for v0.5.0 features.
Tests caching, GPU selection, and rules support.
"""

from src.cracker import HashCracker, crack_hash
import time
from pathlib import Path

def test_cache():
    """Test crack result caching."""
    print("=" * 60)
    print("TEST 1: Crack Result Caching")
    print("=" * 60)
    
    # Test hash (MD5 of "hello")
    test_hash = "5d41402abc4b2a76b9719d911017c592"
    
    # Clear cache database for clean test
    cache_db = Path.home() / '.hashmind' / 'cracking' / 'crack_cache.db'
    if cache_db.exists():
        cache_db.unlink()
        print("✓ Cleared cache database\n")
    
    # First crack (should take time)
    print("First crack attempt (no cache):")
    cracker = HashCracker(use_cache=True)
    start = time.time()
    result1 = cracker.crack(test_hash, "md5_hex", max_time=60)
    time1 = time.time() - start
    
    if result1.success:
        print(f"✓ Cracked: {result1.plaintext}")
        print(f"  Time: {time1:.2f}s\n")
    else:
        print(f"✗ Failed: {result1.error}\n")
        return False
    
    # Second crack (should be instant from cache)
    print("Second crack attempt (should use cache):")
    start = time.time()
    result2 = cracker.crack(test_hash, "md5_hex", max_time=60)
    time2 = time.time() - start
    
    if result2.success:
        print(f"✓ Retrieved: {result2.plaintext}")
        print(f"  Time: {time2:.2f}s")
        print(f"  Speedup: {time1/time2:.1f}x faster!\n")
        
        if "cached" in result2.method.lower():
            print("✓ Cache is working correctly!\n")
            return True
        else:
            print("✗ Cache was not used\n")
            return False
    else:
        print(f"✗ Failed: {result2.error}\n")
        return False


def test_parameters():
    """Test new parameters (without actually running)."""
    print("=" * 60)
    print("TEST 2: New Parameters")
    print("=" * 60)
    
    try:
        # Test with all new parameters
        result = crack_hash(
            "test_hash",
            "md5_hex",
            use_cache=False,
            rules_file="/nonexistent/rules.txt",
            device="1",
            max_time=1
        )
        print("✓ All new parameters accepted\n")
        return True
    except TypeError as e:
        print(f"✗ Parameter error: {e}\n")
        return False


def test_cache_database():
    """Test cache database structure."""
    print("=" * 60)
    print("TEST 3: Cache Database Structure")
    print("=" * 60)
    
    import sqlite3
    
    cache_db = Path.home() / '.hashmind' / 'cracking' / 'crack_cache.db'
    
    if not cache_db.exists():
        print("✗ Cache database not found\n")
        return False
    
    try:
        conn = sqlite3.connect(str(cache_db))
        cursor = conn.cursor()
        
        # Check table structure
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if ('cracks',) in tables:
            print("✓ 'cracks' table exists")
        else:
            print("✗ 'cracks' table not found")
            return False
        
        # Check columns
        cursor.execute("PRAGMA table_info(cracks)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        expected_columns = {
            'hash_value': 'TEXT',
            'plaintext': 'TEXT',
            'hash_type': 'TEXT',
            'method': 'TEXT',
            'timestamp': 'INTEGER'
        }
        
        for col, col_type in expected_columns.items():
            if col in columns:
                print(f"  ✓ Column '{col}' ({col_type})")
            else:
                print(f"  ✗ Column '{col}' missing")
                return False
        
        # Check row count
        cursor.execute("SELECT COUNT(*) FROM cracks")
        count = cursor.fetchone()[0]
        print(f"\n✓ Cache contains {count} entries\n")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Database error: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "hashmind v0.5.0 Feature Test" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = []
    
    # Test 1: Caching
    results.append(("Crack Result Caching", test_cache()))
    
    # Test 2: Parameters
    results.append(("New Parameters", test_parameters()))
    
    # Test 3: Database
    results.append(("Cache Database", test_cache_database()))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! v0.5.0 features are working correctly.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.\n")
        return 1


if __name__ == '__main__':
    exit(main())
