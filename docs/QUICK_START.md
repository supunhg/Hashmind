# hashmind Quick Start Guide

## Installation

```bash
# From PyPI (when published)
pip install hashmind

# Or from GitHub
pip install git+https://github.com/supunhg/hashmind.git

# Or local development
git clone https://github.com/supunhg/hashmind.git
cd hashmind
pip install -e .
```

## Basic Usage

### 1. Identify a Hash

```bash
# Basic identification
hashmind 5d41402abc4b2a76b9719d911017c592
# Output: md5_hex

# Or use short alias
hmind 5d41402abc4b2a76b9719d911017c592
```

### 2. Show Confidence Scores

```bash
hmind -c 5d41402abc4b2a76b9719d911017c592
# Shows all matches with confidence percentages
```

### 3. Crack a Hash

```bash
# Basic cracking
hmind -C 5d41402abc4b2a76b9719d911017c592

# With custom wordlist
hmind -C -w /path/to/rockyou.txt <hash>

# With timeout (seconds)
hmind -C -t 60 <hash>

# GPU selection (for multi-GPU systems)
hmind -C -d 1 <hash>

# With hashcat rules
hmind -C -r /path/to/rules.rule <hash>

# Disable cache
hmind -C --no-cache <hash>
```

### 4. Batch Processing

```bash
# From file
cat hashes.txt | hmind -b

# With confidence scores
cat hashes.txt | hmind -b -c
```

### 5. Check Cracking Tools

```bash
hmind -T
# Shows if hashcat/john are installed
```

## Python API

```python
from hashmind import identify, crack_hash

# Identify hash
result = identify("5d41402abc4b2a76b9719d911017c592")
print(result.top_match())  # md5_hex

# Get all matches with confidence
for match in result.matches:
    print(f"{match['algorithm']}: {match['confidence']:.2%}")

# Crack hash
crack_result = crack_hash(
    "5d41402abc4b2a76b9719d911017c592",
    wordlist="/path/to/wordlist.txt",
    max_time=300
)

if crack_result.success:
    print(f"Cracked: {crack_result.plaintext}")
```

## Supported Hash Types

- **MD Family:** MD5, MD4, MD2
- **SHA Family:** SHA-1, SHA-256, SHA-512, SHA-3
- **Password Hashing:** bcrypt, scrypt, argon2, PBKDF2
- **Databases:** MySQL, PostgreSQL, MSSQL, Oracle
- **Web Frameworks:** Django, WordPress, Joomla
- **Cryptocurrencies:** Bitcoin, Ethereum addresses
- **Archives:** PKZIP, WinZip, RAR, 7-Zip
- **Encodings:** Base64, Hex, Base58
- **And 60+ more!**

## Common Scenarios

### CTF Hash Cracking

```bash
# 1. Identify the hash
hmind hash.txt

# 2. Crack with rockyou
hmind -C -w /usr/share/wordlists/rockyou.txt hash.txt
```

### Forensics Analysis

```bash
# Batch analyze hashes from evidence
cat evidence_hashes.txt | hmind -b -c > analysis.txt
```

### Password Auditing

```bash
# Crack weak passwords with rules
hmind -C -r best64.rule -w common_passwords.txt hash_file.txt
```

## Tips & Tricks

1. **Use cache:** Caching is enabled by default - subsequent cracks of the same hash are instant!

2. **GPU acceleration:** If you have multiple GPUs, use `-d` to select the fastest one

3. **Rules multiply effectiveness:** Use `-r` with a good rules file to dramatically increase crack success

4. **Batch processing:** For many hashes, use `-b` flag for cleaner output

5. **Time limits:** Set realistic `-t` values - some hashes take hours

## Troubleshooting

**Hash not identified:**
- Try with `-c` to see alternative matches
- Hash might be custom or obscure
- Check for extra characters or whitespace

**Cracking fails:**
- Verify hashcat/john installed: `hmind -T`
- Check wordlist exists and is readable
- Try with `-v` for verbose output
- Increase timeout with `-t`

**Import errors:**
```bash
pip install --upgrade hashmind
# Or reinstall
pip uninstall hashmind
pip install hashmind
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for internal details
- See [ROADMAP.md](ROADMAP.md) for upcoming features
- Check [CHANGELOG.md](CHANGELOG.md) for version history
- Read [PACKAGING.md](PACKAGING.md) to publish your own fork

## Getting Help

- **Issues:** https://github.com/supunhg/hashmind/issues
- **Discussions:** https://github.com/supunhg/hashmind/discussions
- **Documentation:** https://github.com/supunhg/hashmind/tree/main/docs
