#!/bin/bash
set -euo pipefail

# Sets up uv (all extras), optional git hooks, and system packages
# (raylib, qhull, xvfb). System packages are installed automatically on apt-based
# systems; elsewhere the required packages are listed for manual installation.

cd "$(dirname -- "${BASH_SOURCE[0]}")"

echo "🚀 Setting up development environment..."

in_git_repo() {
    git rev-parse --is-inside-work-tree >/dev/null 2>&1
}

# Run as root directly, otherwise via sudo when available
SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo &>/dev/null; then
    SUDO="sudo"
fi

# Check if uv is available, install if not
if ! command -v uv &>/dev/null; then
    echo "⚠️  uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    # Verify installation
    if ! command -v uv &>/dev/null; then
        echo "❌ Failed to install uv. Please install manually:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed"
    echo ""
    echo "⬆️  Updating uv ..."
    # Fails for uv installed via pip/brew/conda; those manage updates themselves
    uv self update || echo "⚠️  Skipping uv self update (uv not installed via the standalone installer)"
fi

# On NFS, keep the venv off the shared filesystem unless the shell already chose one
if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ] && [[ "$(findmnt -T . -o FSTYPE -n 2>/dev/null || true)" == nfs* ]]; then
    export UV_PROJECT_ENVIRONMENT="${UV_BASE:-$HOME/.uv-local}/venvs/$(basename "$PWD")"
    echo "📁 NFS detected; using environment at $UV_PROJECT_ENVIRONMENT"
fi

# Install project locally with all extras (pufferlib needs a non-isolated rebuild)
echo ""
echo "📦 Installing project locally with all extras (dev, test)..."
uv sync --all-extras --no-install-package pufferlib
uv sync --all-extras --no-build-isolation-package pufferlib

echo "✅ Project installed with all extras"

# Git hooks (skip when there is no repo — e.g. copied tree)
echo ""
if in_git_repo; then
    echo "🔧 Installing pre-commit hooks..."
    uv run pre-commit install || echo "⚠️  Failed to install pre-commit hooks"
else
    echo "⚠️  No git repository detected; skipping pre-commit install."
    echo "   After \`git init\`, re-run: uv run pre-commit install"
fi

# System packages: raylib (C binding / render), qhull (C binding), xvfb (headless render)
echo ""
echo "📦 Installing system packages (raylib, qhull, xvfb)..."
if command -v apt-get &>/dev/null && { [ -n "$SUDO" ] || [ "$(id -u)" -eq 0 ]; }; then
    $SUDO apt-get update
    $SUDO apt-get install -y libqhull-dev xvfb
    uv run build --install-raylib-only
    echo "✅ System packages installed"
else
    echo "⚠️  apt-get (or root/sudo access) not available; install these manually:"
    echo "   - raylib (https://github.com/raysan5/raylib, headers in /usr/include or /usr/local/include)"
    echo "   - qhull development headers (libqhull-dev / qhull-devel)"
    echo "   - Xvfb (only needed for rendering without a display)"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
