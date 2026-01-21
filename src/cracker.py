"""Hash cracking functionality using external tools (hashcat/john)."""

import subprocess
import shutil
import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
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
    
    Integrates with hashcat and john the ripper for actual cracking.
    Provides a unified interface with retro-styled progress display.
    """
    
    def __init__(self, wordlist: Optional[str] = None):
        """
        Initialize hash cracker.
        
        Args:
            wordlist: Path to wordlist file (optional, uses common passwords if not provided)
        """
        self.wordlist = wordlist
        self.hashcat_path = shutil.which('hashcat')
        self.john_path = shutil.which('john')
        
        # Create temp directory for cracking operations
        self.temp_dir = Path.home() / '.hashmind' / 'cracking'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def is_available(self) -> Dict[str, bool]:
        """Check which cracking tools are available."""
        return {
            'hashcat': self.hashcat_path is not None,
            'john': self.john_path is not None
        }
    
    def crack(self, hash_value: str, hash_type: str, 
              max_time: int = 300, use_rules: bool = False) -> CrackResult:
        """
        Attempt to crack a hash.
        
        Args:
            hash_value: Hash to crack
            hash_type: Detected hash type
            max_time: Maximum time in seconds (default: 300 = 5 minutes)
            use_rules: Apply mutation rules (slower but more effective)
            
        Returns:
            CrackResult with outcome
        """
        # Display retro banner
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
            result = self._crack_with_hashcat(hash_value, hash_type, max_time, use_rules)
            if result.success:
                return result
        
        # Fallback to john the ripper
        if available['john'] and hash_type in JOHN_FORMATS:
            console.print("[dim]Attempting with john the ripper...[/dim]")
            result = self._crack_with_john(hash_value, hash_type, max_time)
            if result.success:
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
                           max_time: int, use_rules: bool) -> CrackResult:
        """Crack using hashcat."""
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
            str(hash_file),
            wordlist
        ]
        
        if use_rules:
            cmd.extend(['-r', '/usr/share/hashcat/rules/best64.rule'])
        
        try:
            # Run with progress indicator
            with self._retro_progress("Cracking with hashcat") as progress:
                task = progress.add_task("Processing...", total=None)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max_time + 10
                )
                
                progress.update(task, completed=True)
            
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
            with self._retro_progress("Cracking with john") as progress:
                task = progress.add_task("Processing...", total=None)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max_time + 10
                )
                
                progress.update(task, completed=True)
            
            # Check if cracked
            show_cmd = [self.john_path, '--show', str(hash_file)]
            show_result = subprocess.run(show_cmd, capture_output=True, text=True)
            
            if show_result.stdout and ':' in show_result.stdout:
                plaintext = show_result.stdout.split(':', 1)[1].split('\n')[0].strip()
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
            if hash_file.exists():
                hash_file.unlink()
    
    def _get_wordlist(self) -> Optional[str]:
        """Get wordlist path, using built-in common passwords if none specified."""
        if self.wordlist and os.path.exists(self.wordlist):
            return self.wordlist
        
        # Try common wordlist locations
        common_locations = [
            '/usr/share/wordlists/rockyou.txt',
            '/usr/share/dict/words',
            '/usr/share/john/password.lst',
        ]
        
        for location in common_locations:
            if os.path.exists(location):
                return location
        
        # Create minimal wordlist if none found
        minimal_wordlist = self.temp_dir / 'common.txt'
        if not minimal_wordlist.exists():
            common_passwords = [
                'password', 'password123', '123456', '12345678', 'qwerty',
                'abc123', 'monkey', '1234567', 'letmein', 'trustno1',
                'dragon', 'baseball', 'iloveyou', 'master', 'sunshine',
                'ashley', 'bailey', 'passw0rd', 'shadow', 'superman',
                'hello', 'welcome', 'admin', 'root', 'toor'
            ]
            minimal_wordlist.write_text('\n'.join(common_passwords))
        
        return str(minimal_wordlist)
    
    def _retro_progress(self, description: str):
        """Create progress indicator."""
        return Progress(
            SpinnerColumn("dots"),
            TextColumn("[dim]{task.description}[/dim]", justify="left"),
            BarColumn(bar_width=40),
            TimeRemainingColumn(),
            console=console,
            transient=True
        )
    
    def _display_success(self, plaintext: str, time_taken: float, method: str):
        """Display success message."""
        console.print()
        console.print(f"[green]✓[/green] Cracked: [bold]{plaintext}[/bold]")
        console.print(f"[dim]  Time: {time_taken:.2f}s | Method: {method}[/dim]")


def crack_hash(hash_value: str, hash_type: Optional[str] = None, 
               wordlist: Optional[str] = None, max_time: int = 300) -> CrackResult:
    """
    Convenience function to crack a hash.
    
    Args:
        hash_value: Hash to crack
        hash_type: Hash type (auto-detected if None)
        wordlist: Path to wordlist
        max_time: Maximum time in seconds
        
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
    
    cracker = HashCracker(wordlist=wordlist)
    return cracker.crack(hash_value, hash_type, max_time)
