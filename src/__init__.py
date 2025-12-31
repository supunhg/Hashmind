"""hashmind - The intelligent hash identifier."""

__version__ = "0.3.0"
__author__ = "Supun Hewagamage"
__description__ = "Intelligent hash/format identification system combining heuristics and ML"

from .identifier import identify, identify_batch, clear_cache, get_cache_info
from .decoder import decode_recursive
from .threshold_tuner import get_tuner, should_report_match

__all__ = ["identify", "identify_batch", "clear_cache", "get_cache_info", 
           "decode_recursive", "get_tuner", "should_report_match"]
