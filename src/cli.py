#!/usr/bin/env python3
"""Command-line interface for hashmind."""

import sys
import argparse
from typing import Optional
from . import identify, __version__
from .cracker import crack_hash, HashCracker


def main(args: Optional[list] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Command-line arguments (for testing)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog='hmind',
        description='🔐 Intelligent hash identification and cracking system',
        epilog='Examples:\n'
               '  hmind 5d41402abc4b2a76b9719d911017c592\n'
               '  hmind -c "$hash"              # show confidence\n'
               '  hmind -C "$hash"              # crack hash\n'
               '  hmind -C -w rockyou.txt "$hash"\n'
               '  hmind -C -r best64.rule "$hash"   # with rules\n'
               '  hmind -C -d 1 "$hash"         # use GPU 1\n'
               '  hmind -T                      # check tools\n'
               '  cat hashes.txt | hmind -b',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Hash or string to identify (reads from stdin if omitted)'
    )
    
    parser.add_argument(
        '-c', '--confidence',
        action='store_true',
        help='Show confidence scores for all matches'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed analysis including metadata'
    )
    
    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help='Process multiple inputs from stdin (one per line)'
    )
    
    parser.add_argument(
        '-C', '--crack',
        action='store_true',
        help='Attempt to crack the hash (requires hashcat or john)'
    )
    
    parser.add_argument(
        '-w', '--wordlist',
        type=str,
        help='Path to wordlist for cracking (optional)'
    )
    
    parser.add_argument(
        '-r', '--rules',
        type=str,
        help='Path to hashcat rules file (optional)'
    )
    
    parser.add_argument(
        '-d', '--device',
        type=str,
        help='GPU device selection for hashcat (e.g., "1" or "1,2")'
    )
    
    parser.add_argument(
        '-t', '--max-time',
        type=int,
        default=300,
        help='Maximum cracking time in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable crack result caching'
    )
    
    parser.add_argument(
        '-T', '--check-tools',
        action='store_true',
        help='Check availability of cracking tools'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    parsed_args = parser.parse_args(args)
    
    # Check tools availability
    if parsed_args.check_tools:
        return check_cracking_tools()
    
    # Handle batch mode
    if parsed_args.batch:
        return batch_mode(parsed_args)
    
    # Get input from argument or stdin
    if parsed_args.input:
        input_string = parsed_args.input
    else:
        if sys.stdin.isatty():
            parser.print_help()
            return 1
        input_string = sys.stdin.read().strip()
    
    # Perform identification
    try:
        result = identify(input_string)
        
        # Cracking mode
        if parsed_args.crack:
            return crack_mode(input_string, result.top_match(), parsed_args)
        
        # Display results
        if parsed_args.verbose:
            print(result)
            print(f"\nMetadata:")
            print(f"  Length: {result.metadata['length']}")
            print(f"  Shannon Entropy: {result.metadata['entropy']['shannon']:.2f}")
            print(f"  Character Set: {result.metadata['charset']}")
        elif parsed_args.confidence:
            if result.matches:
                for match in result.matches:
                    print(f"{match['algorithm']}: {match['confidence']:.2%}")
            else:
                print("No matches found")
        else:
            # Simple mode - just print top match
            top = result.top_match()
            if top:
                print(top)
            else:
                print("unknown")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def check_cracking_tools() -> int:
    """Check and display availability of cracking tools."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    
    console = Console()
    cracker = HashCracker()
    available = cracker.is_available()
    
    table = Table(title="Cracking Tools", box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Tool")
    table.add_column("Status")
    table.add_column("Path", style="dim")
    
    table.add_row(
        "hashcat",
        "[green]✓[/green] Available" if available['hashcat'] else "[red]✗[/red] Not Found",
        cracker.hashcat_path or "N/A"
    )
    table.add_row(
        "john",
        "[green]✓[/green] Available" if available['john'] else "[red]✗[/red] Not Found",
        cracker.john_path or "N/A"
    )
    
    console.print(table)
    
    if not any(available.values()):
        console.print("\n[yellow]Note:[/yellow] Install hashcat or john the ripper for cracking support.")
    
    return 0


def crack_mode(hash_value: str, hash_type: Optional[str], args) -> int:
    """
    Handle hash cracking mode.
    
    Args:
        hash_value: Hash to crack
        hash_type: Detected hash type
        args: CLI arguments
        
    Returns:
        Exit code
    """
    from rich.console import Console
    
    console = Console()
    
    if not hash_type:
        console.print("[red]✗ Could not identify hash type. Cannot crack.[/red]")
        return 1
    
    console.print(f"[cyan]Identified hash type:[/cyan] {hash_type}\n")
    
    result = crack_hash(
        hash_value,
        hash_type,
        wordlist=args.wordlist,
        max_time=args.max_time,
        use_cache=not args.no_cache,
        rules_file=args.rules if hasattr(args, 'rules') else None,
        device=args.device if hasattr(args, 'device') else None
    )
    
    if result.success:
        return 0
    else:
        console.print(f"[red]✗ Cracking failed:[/red] {result.error}")
        return 1


def batch_mode(args) -> int:
    """
    Process multiple inputs from stdin.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            result = identify(line)
            top = result.top_match() or "unknown"
            
            if args.confidence:
                confidence = result.matches[0]['confidence'] if result.matches else 0.0
                print(f"{line}\t{top}\t{confidence:.2%}")
            else:
                print(f"{line}\t{top}")
        
        except Exception as e:
            print(f"{line}\terror\t{e}", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
