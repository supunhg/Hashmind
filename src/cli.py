#!/usr/bin/env python3
"""Command-line interface for hashmind."""

import sys
import argparse
from typing import Optional
from . import identify, __version__


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
        description='Intelligent hash and format identification',
        epilog='Examples:\n'
               '  hmind 5d41402abc4b2a76b9719d911017c592\n'
               '  hmind --confidence "$hash"\n'
               '  cat hashes.txt | hmind --batch',
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
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    parsed_args = parser.parse_args(args)
    
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
