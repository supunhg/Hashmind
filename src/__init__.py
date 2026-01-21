"""hashmind - The intelligent hash identifier."""

__version__ = "0.5.0"
__author__ = "Supun Hewagamage"
__description__ = "Intelligent hash/format identification and cracking system combining heuristics and ML"

from .identifier import identify, identify_batch, clear_cache, get_cache_info
from .decoder import decode_recursive
from .threshold_tuner import get_tuner, should_report_match
from .cracker import crack_hash, HashCracker, CrackResult

__all__ = ["identify", "identify_batch", "clear_cache", "get_cache_info", 
           "decode_recursive", "get_tuner", "should_report_match",
           "crack_hash", "HashCracker", "CrackResult"]
