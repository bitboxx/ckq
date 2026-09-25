#!/bin/bash
# Install ck extension for Cursor

set -e

echo "📦 Building ck extension for Cursor..."

# Install and compile
npm install
npm run compile

# Package
echo "📦 Packaging extension..."
npx vsce package --no-dependencies

# Get the .vsix file
VSIX=$(ls -t *.vsix | head -1)

echo "✅ Built: $VSIX"

# Try to install with cursor CLI
if command -v cursor &> /dev/null; then
    echo "🚀 Installing to Cursor..."
    cursor --install-extension "$VSIX" --force
    echo "✅ Installed! Restart Cursor to activate."
else
    echo "⚠️  'cursor' command not found."
    echo "Install manually with:"
    echo "  cursor --install-extension $VSIX"
    echo ""
    echo "Or add Cursor to PATH:"
    echo "  Cursor → Shell Command: Install 'cursor' command in PATH"
fi
