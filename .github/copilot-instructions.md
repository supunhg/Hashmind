# hashmind AI Agent Instructions

## Project Overview
hashmind is a **multi-layered hash identification and cracking system** combining fast heuristic detection with XGBoost ML classification, plus integrated hashcat/john the ripper support for hash cracking. The architecture prioritizes speed (sub-millisecond detection) while achieving 100% accuracy on 126K training samples across 60+ hash types.

**Version**: 0.4.1 (adds hash cracking capabilities)

## Architecture: 3-Layer Detection Pipeline + Cracking

1. **Heuristic Detection** (`src/core/`) - Deterministic rules (0.18ms avg)
2. **Feature Extraction** (`src/features/`) - 55 features across 4 categories
3. **ML Classification** (`src/ml/`) - XGBoost model with Bayesian confidence fusion
4. **Hash Cracking** (`src/cracker.py`) - Subprocess wrapper for hashcat/john (v0.4.1)

Critical: Always run heuristics first (fast path), only invoke ML for ambiguous cases or when `use_ml=True`.

## Core Design Patterns

### Lazy Loading Pattern
Global singletons are lazy-loaded to avoid startup overhead:
```python
_ml_classifier = None  # Only loaded when needed
_detector = None       # Cached singleton instance
```

### Multi-Layer Matching Priority
In `DetectorPipeline.analyze()`, matches are collected in priority order:
1. Prefix/suffix (highest confidence: bcrypt `$2a$`, JWT `xxx.xxx.xxx`)
2. Length-based (see `EXACT_LENGTHS` dict in `matchers.py`)
3. Regex patterns (UUID, cryptocurrency addresses)
4. Character set analysis (fallback for ambiguous cases)

### Feature Namespacing
Features are prefixed by type to prevent collisions:
- `struct_*`: Structural (length, delimiters, modulo patterns)
- `stat_*`: Statistical (entropy, character frequency, std dev)
- `algo_*`: Algorithmic (base64 validity, compression ratio)

Example: `struct_length_mod_4`, `stat_unique_ratio`, `algo_base64_valid`

### Caching Strategy
- **LRU cache**: 4096 entries on `_cached_identify()` - returns tuples for hashability
- **Result cache**: 21x speedup for repeated queries
- **Feature cache**: Batch processing reuses extractor instances

## Development Workflows

### Adding New Hash Types
1. Update matchers in `src/core/matchers.py`:
   - Add to `EXACT_LENGTHS[length]` for length-based detection
   - Add to `PREFIX_PATTERNS` for prefix-based (e.g., `$argon2$`)
2. Generate training data: `python scripts/generate_training_data.py --count 10000`
3. Retrain model: `python scripts/train_model.py`
4. Model saved to `models/hashmind_model.pkl`

### Testing Workflow
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```
Tests use real hash samples from `tests/test_identifier.py`. Always test both heuristic-only and ML-enhanced paths.

### Training Pipeline (scripts/)
1. `generate_training_data.py`: Generates 126K samples from 10K plaintexts
   - Parallel hash generation using `ThreadPoolExecutor`
   - Output: `samples/training_data.jsonl` (one JSON per line)
2. `train_model.py`: XGBoost training with feature extraction
   - Batch processing (1000 samples/batch) for memory efficiency
   - 80/20 train/test split, stratified sampling
   - Rich progress bars for user feedback

## Project-Specific Conventions

### Return Type Pattern
Use `IdentificationResult` (not raw dicts) for all identification functions:
```python
result = identify("hash")
result.top_match()        # str: most likely algorithm
result.matches            # List[Dict]: all matches with confidence
result.ml_used            # bool: was ML invoked?
```

### Confidence Scoring
- Heuristic confidence: 0.0-1.0 based on specificity (prefix match = 0.9, length match = 0.6)
- ML probability: Softmax output from XGBoost
- Fused confidence: Bayesian combination in `ml/confidence.py`
- Threshold: >90% for immediate return, 70-90% for combining signals

### CLI Design (`src/cli.py`)
- Two entry points: `hashmind` and `hmind` (alias)
- stdin support: `echo "hash" | hmind`
- Short flags: `-C` (crack), `-c` (confidence), `-T` (check tools), `-b` (batch), `-v` (verbose), `-w` (wordlist), `-t` (max-time)
- Long flags: `--crack`, `--confidence`, `--check-tools`, `--batch`, `--verbose`, `--wordlist`, `--max-time`
- Cracking mode: `hmind -C <hash>` (v0.4.1)
- Tool checking: `hmind -T`
- Rich formatting for verbose output

### Cracking Integration (`src/cracker.py`) - NEW v0.4.1
- **HashCracker** class wraps hashcat/john via subprocess
- **Mode mapping**: `HASHCAT_MODES` and `JOHN_FORMATS` dicts map hash types to tool-specific modes
- **Clean UI**: Professional progress bars and status messages
- **Wordlist fallback**: Auto-searches `/usr/share/wordlists/`, creates minimal list if needed
- **Temp management**: Uses `~/.hashmind/cracking/` for temp files
- **Result parsing**: Extracts plaintext from tool-specific output formats

### Match Deduplication
`DetectorPipeline._deduplicate_matches()` keeps highest confidence per algorithm. Critical for cases where multiple matchers trigger on same hash type (e.g., MD5 matches by length AND hex charset).

## Integration Points

### Model Loading
Model path resolution in `MLClassifier._get_default_model_path()`:
1. Check `model_path` parameter
2. Fall back to `models/hashmind_model.pkl` relative to project root
3. Raise `FileNotFoundError` with helpful message if missing

### Feature Extraction Batching
`FeatureExtractor.extract_batch()` uses parallel processing for batches >50:
- ThreadPoolExecutor with max 8 workers (caps CPU usage)
- Falls back to sequential on exception
- Critical for training performance (10x faster)

### Dependencies
- **XGBoost**: Core ML dependency, version >=2.0.0 required
- **Rich**: All console output uses Rich for progress bars and formatting
- **Pandas**: Feature engineering in training pipeline only (not in runtime identification)

## Common Pitfalls

1. **Don't bypass normalization**: Always use `InputNormalizer.normalize()` before matching
2. *rc/cracker.py`: Hash cracking wrapper (subprocess-based, v0.4.1)
- `scripts/`: Training pipeline (not imported by runtime code)
- `models/`: Trained model artifacts (git-ignored, must be trained locally)
- `~/.hashmind/cracking/`: Temp directory for cracking operations not dicts (immutable requirement)
4. **Batch size tuning**: <50 items use sequential, ≥50 use parallel in `extract_batch()`
5. **ML model optional**: Code must gracefully handle missing model file (heuristics-only mode)

## File Organization Logic

- `src/core/`: Detection engine (no ML dependencies)
- `src/features/`: Feature extraction (shared by detection and training)
- `src/ml/`: ML components (lazy-loaded, optional)
- `scripts/`: Training pipeline (not imported by runtime code)
- `models/`: Trained model artifacts (git-ignored, must be trained locally)

## Development Environment

- **Virtual Environment**: `.venv/` - hashmind is installed here in editable mode
- Activate: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)

## Key Commands Reference

```bash
# Installation (editable mode for development)
pip install -e .

# Generate training data
python scripts/generate_training_data.py --count 10000

# Train model
python scripts/train_model.py

# Run tests with coverage
pytest test - Identification
hashmind "5d41402abc4b2a76b9719d911017c592"
echo "hash" | hmind -c
cat hashes.txt | hmind -b -v

# CLI usage - Cracking (v0.4.1)
hmind -C "5d41402abc4b2a76b9719d911017c592"
hmind -C -w rockyou.txt -t 600 "$hash"
hmind -T  # Check hashcat/john availability

# External dependencies for cracking
# Install hashcat: https://hashcat.net/hashcat/
# Install john: https://www.openwall.com/john/
```

## UI/UX Design Patterns (v0.4.1)

### Professional Output
- Clean, minimal progress indicators
- Consistent use of checkmarks (✓/✗) for status
- Subtle color usage (green for success, red for errors, dim for metadata)
- Simple tables using `box.SIMPLE`

### Progress Indicators
- Use `rich.progress.Progress` with minimal styling
- Standard spinner ("dots") instead of fancy animations
- `transient=True` for cleaner output
- Time remaining display for long operations

## Performance Characteristics

- **Heuristic path**: 0.18ms average (always runs first)
- **ML path**: +10ms for feature extraction + classification
- **Cache hit**: 0.008ms (21x faster than heuristic)
- **Batch processing**: 0.24ms per hash (parallelized feature extraction)

When optimizing, focus on heuristic matchers first - they're the hot path.
