"""Main detection pipeline combining all heuristic matchers."""

from typing import List, Dict, Any
from .normalizer import InputNormalizer
from .matchers import Match, LengthMatcher, PrefixSuffixDetector, RegexEngine
from .analyzers import CharacterSetAnalyzer, StructureValidator, EntropyCalculator


class DetectorPipeline:
    """
    Orchestrates all heuristic detection methods.
    
    Combines length matching, prefix/suffix detection, regex patterns,
    character set analysis, and entropy calculations.
    """
    
    def __init__(self):
        """Initialize the detection pipeline."""
        self.normalizer = InputNormalizer()
        self.length_matcher = LengthMatcher()
        self.prefix_suffix_detector = PrefixSuffixDetector()
        self.regex_engine = RegexEngine()
        self.charset_analyzer = CharacterSetAnalyzer()
        self.structure_validator = StructureValidator()
        self.entropy_calculator = EntropyCalculator()
    
    def analyze(self, input_string: str) -> Dict[str, Any]:
        """
        Run complete heuristic analysis.
        
        Args:
            input_string: String to analyze
            
        Returns:
            Dictionary containing:
                - matches: List of Match objects
                - metadata: Additional analysis data
        """
        # Normalize input
        try:
            normalized = self.normalizer.normalize(input_string)
        except ValueError as e:
            return {
                'matches': [],
                'metadata': {'error': str(e)},
                'success': False
            }
        
        # Collect matches from all detectors
        all_matches = []
        
        # Layer 1: Fast structural matches (highest priority)
        prefix_matches = self.prefix_suffix_detector.match(normalized)
        all_matches.extend(prefix_matches)
        
        # Layer 2: Length-based matches
        all_matches.extend(self.length_matcher.match(normalized))
        
        # Layer 3: Regex patterns
        all_matches.extend(self.regex_engine.match(normalized))
        
        # Layer 4: Character set analysis (only if no strong matches)
        if not prefix_matches:
            all_matches.extend(self.charset_analyzer.get_charset_matches(normalized))
        
        # Calculate entropy and other metadata
        entropy_data = self.entropy_calculator.analyze_entropy(normalized)
        charset_data = self.charset_analyzer.analyze(normalized)
        
        # Deduplicate and sort by confidence
        unique_matches = self._deduplicate_matches(all_matches)
        sorted_matches = sorted(unique_matches, key=lambda m: m.confidence, reverse=True)
        
        return {
            'matches': sorted_matches,
            'metadata': {
                'length': len(normalized),
                'entropy': entropy_data,
                'charset': charset_data,
            },
            'success': True
        }
    
    def _deduplicate_matches(self, matches: List[Match]) -> List[Match]:
        """
        Remove duplicate algorithm matches, keeping highest confidence.
        
        Args:
            matches: List of Match objects
            
        Returns:
            Deduplicated list of matches
        """
        seen = {}
        for match in matches:
            if match.algorithm not in seen or match.confidence > seen[match.algorithm].confidence:
                seen[match.algorithm] = match
        
        return list(seen.values())
    
    def quick_identify(self, input_string: str) -> str:
        """
        Quick identification returning most likely algorithm.
        
        Args:
            input_string: String to identify
            
        Returns:
            Most likely algorithm name or "unknown"
        """
        result = self.analyze(input_string)
        
        if result['success'] and result['matches']:
            return result['matches'][0].algorithm
        
        return "unknown"
