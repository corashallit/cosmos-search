#!/usr/bin/env bash
# SessionStart hook: install the package and its archive clients in cloud sessions.
set -euo pipefail
[[ "${CLAUDE_CODE_REMOTE:-}" == "true" ]] || exit 0
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}"
python3 -m pip install --quiet --disable-pip-version-check -e '.[dev]'
