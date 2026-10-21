#!/usr/bin/env bash
# SessionStart hook: install the package and its archive clients in cloud sessions,
# and extend the proxy's CA bundle with current Mozilla roots.
set -euo pipefail
[[ "${CLAUDE_CODE_REMOTE:-}" == "true" ]] || exit 0
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}"
python3 -m pip install --quiet --disable-pip-version-check --upgrade certifi
python3 -m pip install --quiet --disable-pip-version-check -e '.[dev]'

# Some hosts (sciserver.org, skyserver.sdss.org) reach us with their own chain, rooted
# in "emSign Root CA - G1", which the proxy bundle lacks. Proxy bundle + certifi
# covers both re-signed and passed-through hosts.
proxy_ca=/root/.ccr/ca-bundle.crt
if [[ -f "$proxy_ca" ]]; then
  bundle="$HOME/.cache/cosmos-search/ca-bundle.pem"
  mkdir -p "$(dirname "$bundle")"
  cat "$proxy_ca" "$(python3 -c 'import certifi; print(certifi.where())')" > "$bundle"
  if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
    for v in REQUESTS_CA_BUNDLE SSL_CERT_FILE CURL_CA_BUNDLE; do
      echo "export $v=$bundle" >> "$CLAUDE_ENV_FILE"
    done
  fi
fi
