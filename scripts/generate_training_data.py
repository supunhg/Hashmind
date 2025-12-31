#!/usr/bin/env python3
"""Generate synthetic training data for hash identification."""

import hashlib
import base64
import secrets
import json
import uuid
from typing import List, Dict
import sys
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
from rich.panel import Panel
from rich import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

console = Console()


class HashGenerator:
    """Generate hashes of various types for training."""
    
    def generate_plaintexts(self, count: int = 5000) -> List[str]:
        """Generate random plaintexts for hashing."""
        plaintexts = []
        
        common = [
            'password', 'password123', 'admin', 'letmein', 'welcome',
            'monkey', '1234567890', 'qwerty', 'abc123', 'hello',
            'football', 'iloveyou', 'welcome1', 'admin123', 'password1',
            'trustno1', 'dragon', 'master', 'michael', 'sunshine',
            'superman', 'princess', 'starwars', 'shadow', 'cheese'
        ]
        plaintexts.extend(common)
        
        for _ in range(count - len(common)):
            length = secrets.choice([5, 8, 12, 16, 24, 32, 64])
            plaintexts.append(secrets.token_hex(length))
        
        return plaintexts
    
    def generate_md5(self, plaintext: str) -> Dict[str, str]:
        """Generate MD5 hashes in various formats."""
        md5_hash = hashlib.md5(plaintext.encode()).digest()
        
        return {
            'md5_hex': md5_hash.hex(),
            'md5_raw': md5_hash,
            'md5_base64': base64.b64encode(md5_hash).decode().rstrip('='),
        }
    
    def generate_sha_family(self, plaintext: str) -> Dict[str, str]:
        """Generate SHA family hashes."""
        results = {}
        
        for algo_name, algo_func in [
            ('sha1', hashlib.sha1),
            ('sha224', hashlib.sha224),
            ('sha256', hashlib.sha256),
            ('sha384', hashlib.sha384),
            ('sha512', hashlib.sha512),
        ]:
            hash_bytes = algo_func(plaintext.encode()).digest()
            results[f'{algo_name}_hex'] = hash_bytes.hex()
            results[f'{algo_name}_base64'] = base64.b64encode(hash_bytes).decode().rstrip('=')
        
        return results
    
    def generate_bcrypt_like(self, plaintext: str) -> str:
        """Generate bcrypt-like hash (structure only, not real bcrypt)."""
        # Real bcrypt requires bcrypt library, this is just for structure
        # Format: $2a$10$saltsaltsaltsalthashhashhashhashhashhashhash
        cost = '10'
        salt = base64.b64encode(secrets.token_bytes(16)).decode()[:22]
        hash_part = base64.b64encode(secrets.token_bytes(23)).decode()[:31]
        return f'$2a${cost}${salt}{hash_part}'
    
    def generate_mysql_new(self, plaintext: str) -> str:
        """Generate MySQL new format hash."""
        # MySQL new: SHA1(SHA1(password))
        inner = hashlib.sha1(plaintext.encode()).digest()
        outer = hashlib.sha1(inner).hexdigest().upper()
        return f'*{outer}'
    
    def generate_uuid(self) -> str:
        """Generate UUID."""
        return str(uuid.uuid4())
    
    def generate_jwt_like(self) -> str:
        """Generate JWT-like structure."""
        header = base64.urlsafe_b64encode(json.dumps({
            'alg': 'HS256',
            'typ': 'JWT'
        }).encode()).decode().rstrip('=')
        
        payload = base64.urlsafe_b64encode(json.dumps({
            'sub': secrets.token_hex(8),
            'iat': 1640000000
        }).encode()).decode().rstrip('=')
        
        signature = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('=')
        
        return f'{header}.{payload}.{signature}'
    
    def generate_samples(self, count: int = 1000) -> List[Dict]:
        """Generate complete dataset."""
        print(f"Generating {count} training samples...")
        plaintexts = self.generate_plaintexts(count)
        samples = []
        
        for i, plaintext in enumerate(plaintexts):
            if i % 100 == 0:
                print(f"  Progress: {i}/{count}")
            
            # MD5 variants
            md5_hashes = self.generate_md5(plaintext)
            for algo, hash_val in md5_hashes.items():
                if isinstance(hash_val, bytes):
                    continue  # Skip raw bytes
                samples.append({
                    'hash': hash_val,
                    'algorithm': algo,
                    'plaintext': plaintext,
                    'encoding': 'hex' if 'hex' in algo else 'base64'
                })
            
            # SHA family
            sha_hashes = self.generate_sha_family(plaintext)
            for algo, hash_val in sha_hashes.items():
                samples.append({
                    'hash': hash_val,
                    'algorithm': algo,
                    'plaintext': plaintext,
                    'encoding': 'hex' if 'hex' in algo else 'base64'
                })
            
            # Password hashes (every 5th plaintext)
            if i % 5 == 0:
                samples.append({
                    'hash': self.generate_bcrypt_like(plaintext),
                    'algorithm': 'bcrypt',
                    'plaintext': plaintext,
                    'encoding': 'special'
                })
                
                samples.append({
                    'hash': self.generate_mysql_new(plaintext),
                    'algorithm': 'mysql_new',
                    'plaintext': plaintext,
                    'encoding': 'special'
                })
        
        # Add UUIDs
        for _ in range(count // 10):
            samples.append({
                'hash': self.generate_uuid(),
                'algorithm': 'uuid',
                'plaintext': None,
                'encoding': 'special'
            })
        
        # Add JWTs
        for _ in range(count // 10):
            samples.append({
                'hash': self.generate_jwt_like(),
                'algorithm': 'jwt',
                'plaintext': None,
                'encoding': 'base64url'
            })
        
        print(f"Generated {len(samples)} total samples")
        return samples


def main():
    """Generate and save training data."""
    import argparse
    from collections import Counter
    from rich.table import Table
    
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--count', type=int, default=5000, help='Number of base plaintexts')
    parser.add_argument('--output', type=str, default='samples/training_data.jsonl', help='Output file')
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]hashmind Training Data Generator[/bold cyan]",
        box=box.DOUBLE
    ))
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    generator = HashGenerator()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating samples...", total=None)
        samples = generator.generate_samples(args.count)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Generated {len(samples):,} samples")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Saving to {args.output}...", total=None)
        with open(args.output, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Saved to {args.output}")
    
    algo_counts = Counter(s['algorithm'] for s in samples)
    
    table = Table(title="Algorithm Distribution", box=box.ROUNDED)
    table.add_column("Algorithm", style="cyan")
    table.add_column("Samples", justify="right", style="green")
    
    for algo, count in sorted(algo_counts.items()):
        table.add_row(algo, f"{count:,}")
    
    console.print("\n")
    console.print(table)
    
    console.print(Panel.fit(
        f"[bold green]✓ Generated {len(samples):,} training samples![/bold green]",
        box=box.DOUBLE
    ))


if __name__ == '__main__':
    main()
