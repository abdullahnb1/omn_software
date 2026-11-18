#!/bin/bash
# ==============================================================================
# Helper Script: Connect /home/nacib/Desktop/ME407/6-Software as a Git Submodule
# Usage: ./setup_submodule.sh <git-repo-url>
# Example: ./setup_submodule.sh git@github.com:abdullahnb1/ME407-Software.git
# ==============================================================================
set -e

REPO_URL="$1"
ME407_DIR="/home/nacib/Desktop/ME407"
SUBMODULE_PATH="6-Software"

if [ -z "$REPO_URL" ]; then
    echo "Usage: $0 <git-repo-url>"
    echo "Example: $0 git@github.com:abdullahnb1/ME407-Software.git"
    exit 1
fi

cd "$ME407_DIR"

# Remove placeholder .gitkeep if 6-Software only contains .gitkeep
if [ -f "$SUBMODULE_PATH/.gitkeep" ] && [ "$(ls -A "$SUBMODULE_PATH" | wc -l)" -eq 1 ]; then
    echo "[1/3] Removing placeholder $SUBMODULE_PATH/.gitkeep..."
    git rm -f "$SUBMODULE_PATH/.gitkeep" || rm -f "$SUBMODULE_PATH/.gitkeep"
    rmdir "$SUBMODULE_PATH" 2>/dev/null || true
fi

echo "[2/3] Adding $REPO_URL as a Git submodule at $SUBMODULE_PATH..."
git submodule add "$REPO_URL" "$SUBMODULE_PATH"
git submodule update --init --recursive

echo "[3/3] Done! Submodule status:"
git status
echo ""
echo "To save the submodule link in your main ME407 repo, run:"
echo "  cd $ME407_DIR && git commit -m 'Add 6-Software submodule' && git push"
