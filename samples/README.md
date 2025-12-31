# Training Samples Directory

This directory contains training and test data for ML model development.

## Structure

```
samples/
├── synthetic/          # Generated hash samples
│   ├── md5/
│   ├── sha256/
│   ├── bcrypt/
│   └── ...
├── real_world/         # Real-world samples (anonymized)
├── edge_cases/         # Ambiguous and challenging cases
└── validation/         # Hold-out validation set
```

## Data Format

Samples are stored in JSONL format:

```json
{"hash": "5d41402abc4b2a76b9719d911017c592", "algorithm": "md5_hex", "plaintext": "hello", "salt": null}
{"hash": "$2a$10$...", "algorithm": "bcrypt", "plaintext": "password123", "salt": "..."}
```

## Data Generation

See `/scripts/generate_samples.py` for synthetic data generation.

## Privacy

⚠️ **Never commit real passwords or sensitive data.**
- All real-world samples must be anonymized
- Use synthetic data for testing
- Add `samples/private/` to `.gitignore`

---

**Status**: Sample generation scripts will be added in Phase 2.
