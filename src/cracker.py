"""Hash cracking functionality using external tools (hashcat/john)"""

import subprocess
import shutil
import os
import time
import sqlite3
import threading
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.live import Live
from rich.text import Text

console = Console()

# Hash type to hashcat mode mapping
HASHCAT_MODES = {
    'md5_hex': 0,
    'md5_base64': 0,
    'sha1_hex': 100,
    'sha1_base64': 100,
    'sha256_hex': 1400,
    'sha256_base64': 1400,
    'sha512_hex': 1700,
    'sha512_base64': 1700,
    'bcrypt': 3200,
    'ntlm_hex': 1000,
    'mysql': 300,
    'mysql5': 300,
    'sha3_256_hex': 17400,
    'sha3_512_hex': 17600,
    'pkzip': 17200,
    'winzip': 13600,
    'rar': 13000,
    '7zip': 11600
}

# Hash type to john format mapping
JOHN_FORMATS = {
    'md5_hex': 'raw-md5',
    'md5_base64': 'raw-md5',
    'sha1_hex': 'raw-sha1',
    'sha1_base64': 'raw-sha1',
    'sha256_hex': 'raw-sha256',
    'sha256_base64': 'raw-sha256',
    'sha512_hex': 'raw-sha512',
    'sha512_base64': 'raw-sha512',
    'bcrypt': 'bcrypt',
    'ntlm_hex': 'nt',
    'pkzip': 'PKZIP',
    'winzip': 'ZIP',
    'rar': 'RAR',
    '7zip': '7z',
}


class CrackResult:
    """Result from hash cracking attempt."""
    
    def __init__(self, success: bool, plaintext: Optional[str] = None, 
                 time_taken: float = 0.0, method: str = "", error: Optional[str] = None):
        self.success = success
        self.plaintext = plaintext
        self.time_taken = time_taken
        self.method = method
        self.error = error
    
    def __str__(self) -> str:
        if self.success:
            return f"✓ Cracked: {self.plaintext} (took {self.time_taken:.2f}s using {self.method})"
        return f"✗ Failed: {self.error or 'Could not crack hash'}"


class HashCracker:
    """
    Hash cracking interface using external tools.
    
    v0.5.0 Features:
    - SQLite crack result caching
    - GPU device selection  
    - Custom hashcat rules support
    - Progress estimation
    """
    
    def __init__(self, wordlist: Optional[str] = None, use_cache: bool = True):
        """
        Initialize hash cracker.
        
        Args:
            wordlist: Path to wordlist file (optional)
            use_cache: Whether to use crack result caching (default: True)
        """
        self.wordlist = wordlist
        self.use_cache = use_cache
        self.hashcat_path = shutil.which('hashcat')
        self.john_path = shutil.which('john')
        
        # Create temp directory for cracking operations
        self.temp_dir = Path.home() / '.hashmind' / 'cracking'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache database
        if self.use_cache:
            self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        db_path = self.temp_dir / 'crack_cache.db'
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cracks (
                    hash_value TEXT PRIMARY KEY,
                    plaintext TEXT NOT NULL,
                    hash_type TEXT NOT NULL,
                    method TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            console.print(f"[dim yellow]Warning: Could not initialize cache: {e}[/dim yellow]")
            self.use_cache = False
    
    def _check_cache(self, hash_value: str) -> Optional[CrackResult]:
        """Check if hash was previously cracked."""
        if not self.use_cache:
            return None
            
        try:
            db_path = self.temp_dir / 'crack_cache.db'
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                'SELECT plaintext, hash_type, method FROM cracks WHERE hash_value = ?',
                (hash_value,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                plaintext, hash_type, method = row
                console.print(f"[green]✓[/green] Found in cache")
                return CrackResult(
                    success=True,
                    plaintext=plaintext,
                    time_taken=0.0,
                    method=f"{method} (cached)"
                )
        except Exception:
            pass
        
        return None
    
    def _save_to_cache(self, hash_value: str, plaintext: str, hash_type: str, method: str):
        """Save successful crack to cache."""
        if not self.use_cache:
            return
            
        try:
            db_path = self.temp_dir / 'crack_cache.db'
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT OR REPLACE INTO cracks (hash_value, plaintext, hash_type, method, timestamp)
                   VALUES (?, ?, ?, ?, ?)''',
                (hash_value, plaintext, hash_type, method, int(time.time()))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            console.print(f"[dim yellow]Warning: Could not save to cache: {e}[/dim yellow]")
    
    def is_available(self) -> Dict[str, bool]:
        """Check which cracking tools are available."""
        return {
            'hashcat': self.hashcat_path is not None,
            'john': self.john_path is not None
        }
    
    def crack(self, hash_value: str, hash_type: str, 
              max_time: int = 300, use_rules: bool = False,
              rules_file: Optional[str] = None,
              device: Optional[str] = None) -> CrackResult:
        """
        Attempt to crack a hash.
        
        Args:
            hash_value: Hash to crack
            hash_type: Detected hash type
            max_time: Maximum time in seconds (default: 300)
            use_rules: Apply mutation rules (ignored if rules_file is set)
            rules_file: Path to custom hashcat rules file
            device: GPU device selection (e.g., "1" or "1,2")
            
        Returns:
            CrackResult with outcome
        """
        # Check cache first
        cached_result = self._check_cache(hash_value)
        if cached_result:
            return cached_result
        
        # Display header
        self._display_banner(hash_type)
        
        available = self.is_available()
        
        if not any(available.values()):
            return CrackResult(
                success=False,
                error="No cracking tools found. Install hashcat or john the ripper."
            )
        
        # Try hashcat first (faster with GPU)
        if available['hashcat'] and hash_type in HASHCAT_MODES:
            console.print("[dim]Using hashcat (GPU-accelerated)...[/dim]")
            result = self._crack_with_hashcat(
                hash_value, hash_type, max_time, use_rules, rules_file, device
            )
            if result.success:
                self._save_to_cache(hash_value, result.plaintext, hash_type, result.method)
                return result
        
        # Fallback to john the ripper
        if available['john'] and hash_type in JOHN_FORMATS:
            console.print("[dim]Attempting with john the ripper...[/dim]")
            result = self._crack_with_john(hash_value, hash_type, max_time)
            if result.success:
                self._save_to_cache(hash_value, result.plaintext, hash_type, result.method)
                return result
        
        return CrackResult(
            success=False,
            error=f"Hash type '{hash_type}' not supported for cracking"
        )
    
    def _display_banner(self, hash_type: str):
        """Display cracking header."""
        console.print(f"[bold]Hash Cracking[/bold] | Type: [cyan]{hash_type}[/cyan]")
        console.print()
    
    def _crack_with_hashcat(self, hash_value: str, hash_type: str, 
                           max_time: int, use_rules: bool,
                           rules_file: Optional[str] = None,
                           device: Optional[str] = None) -> CrackResult:
        """Crack using hashcat with progress tracking."""
        start_time = time.time()
        
        # Create hash file
        hash_file = self.temp_dir / 'target.hash'
        hash_file.write_text(hash_value)
        
        # Get wordlist
        wordlist = self._get_wordlist()
        if not wordlist:
            return CrackResult(success=False, error="No wordlist available")
        
        # Build hashcat command
        mode = HASHCAT_MODES[hash_type]
        cmd = [
            self.hashcat_path,
            '-m', str(mode),
            '-a', '0',  # Dictionary attack
            '--quiet',
            '--potfile-disable',
            '--runtime', str(max_time),
        ]
        
        # Add GPU device selection
        if device:
            cmd.extend(['-d', device])
        
        # Add rules support
        if rules_file:
            if not Path(rules_file).exists():
                console.print(f"[dim yellow]Warning: Rules file not found: {rules_file}[/dim yellow]")
            else:
                cmd.extend(['-r', rules_file])
        elif use_rules:
            # Try common rules file locations
            common_rules = [
                '/usr/share/hashcat/rules/best64.rule',
                '/usr/local/share/hashcat/rules/best64.rule',
            ]
            for rules_path in common_rules:
                if Path(rules_path).exists():
                    cmd.extend(['-r', rules_path])
                    break
        
        cmd.extend([str(hash_file), wordlist])
        
        try:
            # Run with progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Cracking...", total=None)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max_time + 10
                )
                
                progress.update(task, completed=100)
            
            # Parse output
            if result.returncode == 0 and result.stdout:
                # Hashcat shows cracked hashes as hash:plaintext
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        plaintext = line.split(':', 1)[1].strip()
                        time_taken = time.time() - start_time
                        
                        self._display_success(plaintext, time_taken, "hashcat")
                        return CrackResult(
                            success=True,
                            plaintext=plaintext,
                            time_taken=time_taken,
                            method="hashcat"
                        )
            
            return CrackResult(success=False, error="No match found in wordlist")
            
        except subprocess.TimeoutExpired:
            return CrackResult(success=False, error=f"Timeout after {max_time}s")
        except Exception as e:
            return CrackResult(success=False, error=f"Hashcat error: {str(e)}")
        finally:
            # Cleanup
            if hash_file.exists():
                hash_file.unlink()
    
    def _crack_with_john(self, hash_value: str, hash_type: str, max_time: int) -> CrackResult:
        """Crack using john the ripper."""
        start_time = time.time()
        
        # Create hash file in john format
        hash_file = self.temp_dir / 'target.john'
        
        # John expects specific formats
        if hash_type in ['md5_hex', 'sha1_hex', 'sha256_hex', 'sha512_hex']:
            hash_file.write_text(f"user:{hash_value}")
        else:
            hash_file.write_text(hash_value)
        
        # Get wordlist
        wordlist = self._get_wordlist()
        
        # Build john command
        cmd = [
            self.john_path,
            '--format=' + JOHN_FORMATS[hash_type],
            '--wordlist=' + (wordlist or '/usr/share/john/password.lst'),
            str(hash_file)
        ]
        
        try:
            # Run with progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Cracking...", total=None)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max_time + 10
                )
                
                progress.update(task, completed=100)
            
            # Parse output - john shows results with --show
            show_cmd = [self.john_path, '--show', str(hash_file)]
            show_result = subprocess.run(show_cmd, capture_output=True, text=True)
            
            if show_result.stdout and ':' in show_result.stdout:
                plaintext = show_result.stdout.split(':')[1].strip()
                time_taken = time.time() - start_time
                
                self._display_success(plaintext, time_taken, "john")
                return CrackResult(
                    success=True,
                    plaintext=plaintext,
                    time_taken=time_taken,
                    method="john"
                )
            
            return CrackResult(success=False, error="No match found in wordlist")
            
        except subprocess.TimeoutExpired:
            return CrackResult(success=False, error=f"Timeout after {max_time}s")
        except Exception as e:
            return CrackResult(success=False, error=f"John error: {str(e)}")
        finally:
            # Cleanup
            if hash_file.exists():
                hash_file.unlink()
    
    def _get_wordlist(self) -> Optional[str]:
        """Get wordlist path."""
        if self.wordlist and Path(self.wordlist).exists():
            return self.wordlist
        
        # Try common wordlist locations
        common_wordlists = [
            '/usr/share/wordlists/rockyou.txt',
            '/usr/share/dict/words',
            '/usr/share/wordlists/common.txt',
        ]
        
        for wl in common_wordlists:
            if Path(wl).exists():
                return wl
        
        # Create minimal wordlist as fallback
        fallback_wl = self.temp_dir / 'common_passwords.txt'
        if not fallback_wl.exists():
            fallback_wl.write_text('\n'.join([
                'password', '123456', '123456789', 'qwerty', 'abc123',
                'password123', '12345678', '111111', '1234567', 'admin',
                'letmein', 'welcome', 'monkey', '1234567890', 'dragon'
            ]))
        
        return str(fallback_wl)
    
    def _display_success(self, plaintext: str, time_taken: float, method: str):
        """Display success message."""
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_row("[green]✓[/green] Cracked:", f"[bold]{plaintext}[/bold]")
        if time_taken > 0:
            table.add_row("", f"[dim]{time_taken:.2f}s using {method}[/dim]")
        console.print(table)
        console.print()


def crack_hash(hash_value: str, hash_type: Optional[str] = None, 
               wordlist: Optional[str] = None, max_time: int = 300,
               use_cache: bool = True, rules_file: Optional[str] = None,
               device: Optional[str] = None) -> CrackResult:
    """
    Convenience function to crack a hash.
    
    Args:
        hash_value: Hash to crack
        hash_type: Hash type (auto-detected if None)
        wordlist: Path to wordlist
        max_time: Maximum time in seconds
        use_cache: Whether to use crack result caching
        rules_file: Path to hashcat rules file
        device: GPU device selection
        
    Returns:
        CrackResult
        
    Example:
        >>> from hashmind import crack_hash
        >>> result = crack_hash("5d41402abc4b2a76b9719d911017c592")
        >>> if result.success:
        ...     print(f"Cracked: {result.plaintext}")
    """
    # Auto-detect hash type if not provided
    if not hash_type:
        from .identifier import identify
        detection = identify(hash_value)
        hash_type = detection.top_match()
        
        if not hash_type:
            return CrackResult(success=False, error="Could not identify hash type")
    
    cracker = HashCracker(wordlist=wordlist, use_cache=use_cache)
    return cracker.crack(hash_value, hash_type, max_time, 
                        rules_file=rules_file, device=device)
