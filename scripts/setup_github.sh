#!/usr/bin/env bash
# Install the GitHub CLI (if needed) and check the login state.
#
# The GitHub CLI is a single static binary; no package manager is required.
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
    echo "==> installing the GitHub CLI"
    ARCH="$(uname -m)"
    case "$ARCH" in
        arm64) PKG="cli_macOS_arm64.zip" ;;
        *)     PKG="cli_macOS_amd64.zip" ;;
    esac
    URL="https://github.com/cli/cli/releases/latest/download/${PKG}"
    TMP="$(mktemp -d)"
    curl -fsSL "$URL" -o "$TMP/gh.zip"
    unzip -q "$TMP/gh.zip" -d "$TMP"
    # Try the usual install location; fall back to ~/.local/bin when sudo is
    # not available without a password.
    if sudo -n true 2>/dev/null; then
        sudo install -m 0755 "$TMP"/gh_*/bin/gh /usr/local/bin/gh
    else
        mkdir -p "$HOME/.local/bin"
        install -m 0755 "$TMP"/gh_*/bin/gh "$HOME/.local/bin/gh"
        echo "installed to $HOME/.local/bin -- add it to your PATH:"
        echo '    export PATH="$HOME/.local/bin:$PATH"'
    fi
    rm -rf "$TMP"
fi

echo "==> gh version: $(gh --version | head -1)"
if gh auth status >/dev/null 2>&1; then
    echo "==> already authenticated:"
    gh auth status
else
    echo "==> not authenticated. Run this next (it opens a browser):"
    echo
    echo "    gh auth login --hostname github.com --git-protocol https --web"
    echo
    echo "then re-run this script to confirm."
fi
