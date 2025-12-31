# Architecture

## Overview

hashmind is a multi-layered hash identification system combining deterministic heuristics with machine learning for high-accuracy detection of 60+ hash types and formats.

## System Design

```
┌──────────────┐
│ Input String │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Normalization        │
│ • Trim whitespace    │
│ • Validate encoding  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ Heuristic Detection (Layer 1)   │
│ • Length matching                │
│ • Prefix/suffix patterns         │
│ • Regex validation               │
│ • Character set analysis         │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ Feature Extraction (Layer 2)    │
│ • Statistical: entropy, charset  │
│ • Structural: length, delimiters │
│ • Lexical: case patterns         │
│ • Semantic: format hints         │
│ • Total: 55 features             │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ ML Classification (Layer 3)     │
│ • XGBoost model                  │
│ • 16 hash types                  │
│ • 100% test accuracy             │
│ • 126K training samples          │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ Confidence Fusion                │
│ • Bayesian combination           │
│ • Adaptive thresholds            │
│ • Calibrated probabilities       │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Ranked Results       │
│ • Top match          │
│ • Confidence scores  │
│ • Alternative matches│
└──────────────────────┘
```

## Core Components

### 1. Heuristic Matchers

Fast rule-based detection with microsecond latency:

**Length Matcher** - Maps exact lengths to hash types
- MD5: 32 chars (hex), 22 chars (base64)
- SHA-256: 64 chars (hex), 43 chars (base64)
- Character set validation for disambiguation

**Prefix Matcher** - Identifies formats by prefixes
- bcrypt: `$2a$`, `$2b$`, `$2y$`
- MySQL: `*` (followed by 40 hex chars)
- JWT: 3-part base64url with dots

**Regex Matcher** - Pattern-based detection
- UUID: 8-4-4-4-12 hex pattern with hyphens
- Cryptocurrency addresses: Base58/Bech32 patterns
- Database-specific formats

### 2. Feature Extractor

Extracts 55 features across 4 categories:

**Statistical Features (12)**
- Shannon entropy
- Character frequency distribution
- Unique character ratio
- Standard deviation of char codes

**Structural Features (18)**
- String length
- Length modulo 4/8
- Delimiter counts (dots, dashes, slashes, dollars)
- Special character presence

**Lexical Features (15)**
- Uppercase/lowercase presence
- Digit presence
- Alphanumeric ratio
- Character set composition

**Semantic Features (10)**
- Base64 validity score
- Hex encoding likelihood
- Compression ratio
- Format-specific hints

### 3. ML Classifier

**Model**: XGBoost (Gradient Boosted Trees)
- Objective: Multi-class softmax
- Classes: 16 hash types
- Max depth: 8
- Learning rate: 0.1
- Estimators: 100

**Training Data**:
- 126,000 samples
- 100,800 training / 25,200 test
- Balanced distribution across classes
- Generated from 10,000 plaintexts

**Performance**:
- Test accuracy: 100%
- Validation loss: ~0.002
- Inference time: ~10ms per hash

**Top Features**:
1. stat_min_char_freq (10.3%)
2. stat_std_dev (8.5%)
3. struct_length_mod_4 (8.4%)
4. struct_has_dots (7.3%)
5. stat_unique_ratio (6.4%)

### 4. Confidence Fusion

Combines heuristic and ML predictions:

```python
final_confidence = bayesian_fusion(
    heuristic_confidence,
    ml_probability,
    prior_distribution
)
```

**Adaptive Thresholds**:
- High confidence (>90%): Return immediately
- Medium (70-90%): Combine signals
- Low (<70%): Flag as uncertain

### 5. Caching Layer

**LRU Cache**:
- Capacity: 1000 entries
- Speedup: 21x for repeated queries
- Thread-safe implementation
- Automatic eviction

### 6. Recursive Decoder

Handles multi-layer encodings:

**Supported Encodings**:
- Base64 / Base64URL
- Hex encoding
- URL encoding

**Features**:
- Max depth: 10 levels
- Loop detection
- Hash preservation (prevents over-decoding)
- Encoding chain tracking

**Example**:
```
base64(hex(url(hash))) → hash
Chain: base64 → hex → url
```

## Data Flow

### Identification Pipeline

```python
1. Input: "5d41402abc4b2a76b9719d911017c592"

2. Normalization: 
   - Trim: "5d41402abc4b2a76b9719d911017c592"
   - Valid: UTF-8, printable

3. Heuristic Detection:
   - Length: 32 chars → [md5_hex, md4_hex, ntlm_hex]
   - Charset: All hex → Boost md5_hex to 75%

4. Feature Extraction:
   - Entropy: 3.95 bits/char
   - Charset: hexadecimal only
   - Length mod 4: 0
   - ...55 features total...

5. ML Classification:
   - Features → XGBoost
   - Prediction: md5_hex (89.9%)

6. Confidence Fusion:
   - Heuristic: 75% md5_hex
   - ML: 89.9% md5_hex
   - Fused: 89.9% md5_hex

7. Output:
   - Top: md5_hex (89.9%)
   - Alternatives: md4_hex (30%), ntlm_hex (30%)
```

### Training Pipeline

```python
1. Generate Plaintexts:
   - 10,000 random strings
   - Various lengths (5-64 chars)

2. Generate Hashes:
   - MD5, SHA-1, SHA-256, SHA-384, SHA-512
   - bcrypt, UUID, JWT
   - Multiple encodings (hex, base64)
   - Total: 126,000 samples

3. Extract Features:
   - 55 features per sample
   - Normalize numeric features
   - Encode categorical features

4. Train XGBoost:
   - 80/20 train/test split
   - Stratified sampling
   - Early stopping
   - Cross-validation

5. Validate:
   - Test accuracy: 100%
   - Per-class metrics
   - Feature importance analysis

6. Save Model:
   - Pickle format
   - Includes label encoder
   - Feature names
   - Metadata
```

## Performance Optimizations

### Speed
- **Heuristics first**: 0.18ms average
- **Cache hits**: 0.008ms (21x faster)
- **Batch processing**: 0.24ms per hash
- **Early exit**: High-confidence matches skip ML

### Memory
- **Feature caching**: Reuse for similar inputs
- **Model lazy loading**: Load ML only when needed
- **LRU eviction**: Bounded memory usage

### Accuracy
- **Ensemble approach**: Heuristics + ML
- **Calibrated probabilities**: Reliable confidence scores
- **Character set filtering**: Reduce false positives

## Extensibility

### Adding New Hash Types

1. **Update Matchers** (`src/core/matchers.py`):
```python
EXACT_LENGTHS[40].append('new_hash_40')
```

2. **Generate Training Data** (`scripts/generate_training_data.py`):
```python
def generate_new_hash(self, plaintext):
    return {'new_hash': hash_function(plaintext)}
```

3. **Retrain Model**:
```bash
python scripts/generate_training_data.py --count 10000
python scripts/train_model.py
```

### Adding New Features

1. **Define Feature** (`src/features/extractor.py`):
```python
def extract_new_feature(self, value):
    return {'new_feature': calculation(value)}
```

2. **Retrain Model** with new features

## Dependencies

**Core**:
- Python 3.8+

**ML**:
- xgboost >= 2.0.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0

**UI**:
- rich >= 13.0.0

## Directory Structure

```
hashmind/
├── src/
│   ├── __init__.py
│   ├── identifier.py       # Main API
│   ├── decoder.py          # Recursive decoder
│   ├── threshold_tuner.py  # Adaptive thresholds
│   ├── core/
│   │   ├── detector.py     # Detection pipeline
│   │   ├── matchers.py     # Heuristic matchers
│   │   └── normalizer.py   # Input normalization
│   ├── features/
│   │   └── extractor.py    # Feature extraction (55 features)
│   └── ml/
│       ├── classifier.py   # XGBoost wrapper
│       └── confidence.py   # Bayesian fusion
├── scripts/
│   ├── generate_training_data.py
│   └── train_model.py
├── models/
│   └── hashmind_model.pkl
├── samples/
│   └── training_data.jsonl
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
└── CHANGELOG.md
```

## Future Enhancements

1. **More Hash Types**: Add BLAKE3, Argon2id variants
2. **Deep Learning**: Transformer model for sequence patterns
3. **Hash Cracking**: Integrate with hashcat/john
4. **Format Conversion**: Convert between hex/base64/raw
5. **Web Interface**: REST API and web UI (Phase 4)

---

**Version**: 0.3.0  
**Author**: Supun Hewagamage ([@supunhg](https://github.com/supunhg))  
**License**: MIT
