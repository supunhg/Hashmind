# Changelog

All notable changes to hashmind.

## [0.4.0] - 2026-01-01

### Performance Improvements
- **⚡ 5-10x Faster Feature Extraction**: Pre-compiled regex patterns, LRU caching (2048 entropy, 1024 compression)
- **⚡ Parallel Processing**: ThreadPoolExecutor for batch operations (>50 features, >100 identifications)
- **⚡ Optimized ML**: Batch prediction, 4096-item result cache, better XGBoost hyperparameters
- **⚡ Faster Training**: Parallel feature extraction in 1000-item batches

### ML Model Enhancements
- **Better XGBoost Config**: 200 trees (was 100), depth 10 (was 8), learning rate 0.05 (was 0.1)
- **Advanced Tuning**: L1/L2 regularization, histogram tree method, all CPU cores
- **Smarter Ensemble**: Higher subsample (0.85), more patience (15 rounds early stopping)

### Code Quality
- **Optimized Algorithms**: zlib level 6 (faster, good enough vs level 9)
- **Module-level Patterns**: Compiled regex for BASE64, HEX, UUID, JWT, etc.
- **Function Caching**: `@functools.lru_cache` on expensive operations
- **Rich Progress**: Better progress bars for batch operations

### Already Working
- ✓ stdin support: `echo "hash" | hashmind` (was already implemented in v0.3.0)

### Technical Details
- Result cache: 1000 → 4096 entries
- Entropy cache: 2048 entries (new)
- Compression cache: 1024 entries (new)
- Parallel threshold: 50 features, 100 identifications
- Thread pool: min(CPU cores, 8) workers

## [0.3.0] - 2025-12-31

### Added
- **Massive Training Data Expansion**: 126,000 samples (10x increase)
- **Rich Terminal Output**: Beautiful progress bars and tables for training
- **Production Optimizations**: Clean code, removed AI comments
- **Comprehensive Documentation**: Rewritten README and ARCHITECTURE

### Changed
- Training data: 12,600 → 126,000 samples
- Model accuracy: Maintained 100% on larger dataset
- Terminal output: Plain text → Rich formatted
- Default training count: 1,000 → 10,000 base plaintexts

### Improved
- Code quality: Removed AI-style comments
- Documentation: Concise and professional
- Training UX: Rich progress bars and status tables
- Model robustness: 10x more training data

### Removed
- Unnecessary test files
- Summary documents (not needed for GitHub)
- Demo script
- Old verbose training output

## [0.2.0] - 2024

### Added
- XGBoost ML classification
- Feature extraction (55 features)
- Training data generation
- 12,600 training samples
- 100% test accuracy

## [0.1.0] - 2024

### Added
- Heuristic detection (60+ hash types)
- CLI interface
- Python API
- LRU caching (21x speedup)
- Recursive decoder
- Active learning framework

---

**Author**: Supun Hewagamage ([@supunhg](https://github.com/supunhg))

