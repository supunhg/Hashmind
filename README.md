# hashmind

**Intelligent hash and format identification using machine learning**

hashmind combines fast heuristic detection with XGBoost classification to identify 60+ hash types, cryptographic algorithms, and encoded formats with high accuracy.

## Features

- 🚀 **Fast Detection** - Sub-millisecond identification (0.18ms average)
- 🧠 **ML-Enhanced** - 100% accuracy with XGBoost on 126K training samples  
- 🔍 **60+ Hash Types** - MD5, SHA families, bcrypt, JWT, cryptocurrencies, databases
- 📊 **Confidence Scores** - Calibrated probabilities for each match
- ⚡ **High Performance** - 5-10x faster with caching, parallel batch processing
- 🔄 **Recursive Decoding** - Handle complex encoding chains
- 📥 **stdin Support** - Pipe input directly: `echo "hash" | hashmind`

## Performance

| Metric | Result |
|--------|--------|
| Feature extraction | 5-10x faster (v0.4.0) |
| Result cache | 4096 entries |
| Parallel batches | >100 items |
| ML accuracy | 100% |
| Training speed | 10x faster (parallel) |

## Installation

```bash
git clone https://github.com/supunhg/hashmind.git
cd hashmind
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

### Command Line

```bash
# Basic identification
hashmind 5d41402abc4b2a76b9719d911017c592
# Output: md5_hex

# Or use the short alias
hmind 5d41402abc4b2a76b9719d911017c592

# stdin support (pipe input)
echo "5d41402abc4b2a76b9719d911017c592" | hmind

# Show confidence scores
hmind --confidence 5d41402abc4b2a76b9719d911017c592

# Batch processing
cat hashes.txt | hmind --batch

# Verbose output
hmind --verbose '$2a$10$N9qo8uLOickgx2ZMRZoMye'
```

### Python API

```python
from hashmind import identify

result = identify("5d41402abc4b2a76b9719d911017c592", use_ml=True)
print(result.top_match())  # md5_hex
print(result.matches[0]['confidence'])  # 0.899

from hashmind import identify_batch
hashes = ["5d41402abc...", "550e8400-e29b...", "$2a$10$N9qo..."]
results = identify_batch(hashes, use_ml=True)

from hashmind import decode_recursive
result = decode_recursive("NWQ0MTQwMmFiYzRiMmE3NmI5NzE5ZDkxMTAxN2M1OTI=")
print(result.final_value)  # Original hash
print(result.get_chain())  # base64
```

## Supported Hash Types (60+)

**Cryptographic** (18): MD5, SHA-1/224/256/384/512, SHA-3, BLAKE2, RIPEMD-160

**Passwords** (15): bcrypt, scrypt, Argon2, PBKDF2, Unix crypt, LDAP, NTLM

**Databases** (8): MySQL, PostgreSQL, Oracle, MSSQL, Django

**Cryptocurrency** (5): Bitcoin, Ethereum, Litecoin, Ripple, Monero

**Formats** (8): JWT, UUID, API keys, session tokens

**File Hashes** (4): SSDeep, CRC32, Adler32

**Encodings** (6): Base64, Hex, URL, Base32, Base58

## Architecture

```
hashmind/
├── src/
│   ├── core/           # Heuristic matchers
│   ├── features/       # Feature extraction (55 features)
│   ├── ml/             # XGBoost classifier
│   ├── identifier.py   # Main API
│   └── decoder.py      # Recursive decoder
├── scripts/
│   ├── generate_training_data.py
│   └── train_model.py
└── models/
    └── hashmind_model.pkl
```

## Training Your Own Model

```bash
python scripts/generate_training_data.py --count 10000
python scripts/train_model.py
```

This generates 126,000 samples across 16 hash types and trains an XGBoost model with 100% test accuracy.

## License

MIT

## Author

Supun Hewagamage ([@supunhg](https://github.com/supunhg))
