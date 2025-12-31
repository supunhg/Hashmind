# hashmind

**Intelligent hash and format identification using machine learning**

hashmind combines fast heuristic detection with XGBoost classification to identify 60+ hash types, cryptographic algorithms, and encoded formats with high accuracy.

## Features

- 🚀 **Fast Detection** - Sub-millisecond identification (0.18ms average)
- 🧠 **ML-Enhanced** - 100% accuracy with XGBoost on 126K training samples  
- 🔍 **60+ Hash Types** - MD5, SHA families, bcrypt, JWT, cryptocurrencies, databases
- 📊 **Confidence Scores** - Calibrated probabilities for each match
- ⚡ **High Performance** - 21x cache speedup, efficient batch processing
- 🔄 **Recursive Decoding** - Handle complex encoding chains

## Performance

| Metric | Result |
|--------|--------|
| Heuristic speed | 0.18ms |
| Cached speed | 0.008ms |
| Cache speedup | 21x |
| Batch processing | 0.24ms/hash |
| ML accuracy | 100% |

## Installation

```bash
git clone https://github.com/supunhg/hashmind.git
cd hashmind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

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

### Command Line

```bash
python -m hashmind <hash> [--ml] [--verbose] [--decode]

python -m hashmind 5d41402abc4b2a76b9719d911017c592 --ml
python -m hashmind --decode "NWQ0MTQwMmFiYzRiMmE3NmI5NzE5ZDkxMTAxN2M1OTI="
cat hashes.txt | python -m hashmind --batch --ml
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
