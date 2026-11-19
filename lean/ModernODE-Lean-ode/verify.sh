#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export LC_ALL=C
export LANG=C
export PATH="$HOME/.elan/bin:$PATH"
if ! command -v lake >/dev/null 2>&1; then
  echo "Lean's elan/lake installation is required. See README.md." >&2
  exit 1
fi
lake build
lake env lean Audit.lean
