#!/bin/bash
set -e

VERSION="0.5.0"
PACKAGE_NAME="hashmind"
ARCH="all"
BUILD_DIR="debian-build"

echo "Building hashmind .deb package v${VERSION}..."

# Clean previous builds
rm -rf ${BUILD_DIR} *.deb

# Create directory structure
mkdir -p ${BUILD_DIR}/DEBIAN
mkdir -p ${BUILD_DIR}/opt/hashmind
mkdir -p ${BUILD_DIR}/usr/local/bin

# Copy source files
cp -r src ${BUILD_DIR}/opt/hashmind/
cp -r scripts ${BUILD_DIR}/opt/hashmind/
cp -r models ${BUILD_DIR}/opt/hashmind/
cp requirements.txt ${BUILD_DIR}/opt/hashmind/
cp LICENSE ${BUILD_DIR}/opt/hashmind/
cp README.md ${BUILD_DIR}/opt/hashmind/

# Create control file
cat > ${BUILD_DIR}/DEBIAN/control << EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Depends: python3 (>= 3.8), python3-pip, python3-venv
Maintainer: Supun Hewagamage <supunhg@users.noreply.github.com>
Description: Intelligent hash identification and cracking system
 hashmind combines fast heuristic detection with XGBoost classification
 to identify 60+ hash types with integrated hashcat/john support.
 Features include GPU selection, crack caching, and custom rules.
EOF

# Create postinst script (runs after installation)
cat > ${BUILD_DIR}/DEBIAN/postinst << 'EOF'
#!/bin/bash
set -e

INSTALL_DIR="/opt/hashmind"
VENV_DIR="${INSTALL_DIR}/.venv"

echo "Setting up hashmind virtual environment..."

# Detect system Python version
PYTHON_BIN=$(which python3)
PYTHON_VERSION=$($PYTHON_BIN --version | cut -d' ' -f2 | cut -d'.' -f1,2)

echo "Using Python ${PYTHON_VERSION} from ${PYTHON_BIN}"

# Create venv with system Python
cd ${INSTALL_DIR}
${PYTHON_BIN} -m venv ${VENV_DIR}

# Install dependencies
${VENV_DIR}/bin/pip install --upgrade pip > /dev/null 2>&1
${VENV_DIR}/bin/pip install -r requirements.txt > /dev/null 2>&1

# Create wrapper scripts
cat > /usr/local/bin/hashmind << 'WRAPPER'
#!/bin/bash
/opt/hashmind/.venv/bin/python /opt/hashmind/src/cli.py "$@"
WRAPPER

cat > /usr/local/bin/hmind << 'WRAPPER'
#!/bin/bash
/opt/hashmind/.venv/bin/python /opt/hashmind/src/cli.py "$@"
WRAPPER

chmod +x /usr/local/bin/hashmind
chmod +x /usr/local/bin/hmind

echo "hashmind v0.5.0 installed successfully!"
echo "Run 'hashmind --help' or 'hmind --help' to get started"
EOF

chmod 755 ${BUILD_DIR}/DEBIAN/postinst

# Create prerm script (runs before removal)
cat > ${BUILD_DIR}/DEBIAN/prerm << 'EOF'
#!/bin/bash
set -e

rm -f /usr/local/bin/hashmind
rm -f /usr/local/bin/hmind

exit 0
EOF

chmod 755 ${BUILD_DIR}/DEBIAN/prerm

# Build the package
dpkg-deb --build ${BUILD_DIR} ${PACKAGE_NAME}_${VERSION}_${ARCH}.deb

# Cleanup
rm -rf ${BUILD_DIR}

echo ""
echo "✓ Package built successfully: ${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
echo ""
echo "Install with:"
echo "  sudo dpkg -i ${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
echo ""
echo "Uninstall with:"
echo "  sudo dpkg -r ${PACKAGE_NAME}"
