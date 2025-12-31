# Test vectors for hash identification

This directory contains test vectors for various hash algorithms and formats.

## Structure

- `hash_vectors.json` - Known hash samples with expected results
- `edge_cases.json` - Edge cases and ambiguous inputs
- `real_world.json` - Real-world examples from various sources

## Format

```json
{
  "algorithm": "md5_hex",
  "samples": [
    {
      "input": "5d41402abc4b2a76b9719d911017c592",
      "plaintext": "hello",
      "expected": "md5_hex",
      "confidence_min": 0.7
    }
  ]
}
```
