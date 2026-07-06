#!/usr/bin/env bash
# launch_claude.sh — start Claude Code pointed at the local oMLX endpoint.
#
# Encodes the launch sequence + the gotchas logged in docs/journal.md:
#   - unset ANTHROPIC_API_KEY first (a real cloud key conflicts with the
#     launcher's auth token and silently sends traffic to the cloud)
#   - verify oMLX is actually serving before launching (saves a confusing
#     "connection refused" mid-session)
#   - pin the model deliberately (the dashboard default has drifted to Qwen
#     before; D005 says the trusted default is Gemma 31B + MTP)
#
# Usage:
#   ./launch_claude.sh                 # uses the default model below
#   ./launch_claude.sh MODEL_NAME      # override for one launch
#
# In-session gotcha (not scriptable): if Claude Code resolves the wrong model
# anyway, type /model inside it — a stale stored preference can override env
# vars. This script can't fix that; the /model command inside the session can.

set -euo pipefail

ENDPOINT="${OMLX_ENDPOINT:-http://127.0.0.1:8000}"
AUTH="${OMLX_AUTH:-claude}"
# D005 trusted default. Override by passing a model name as $1.
MODEL="${1:-gemma-4-31B-it-MLX-6bit}"

echo "=== launch_claude — local oMLX ==="
echo "endpoint: $ENDPOINT"
echo "model:    $MODEL"
echo ""

# 1. Clear any cloud key so the launcher's token is the one used.
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "→ unsetting ANTHROPIC_API_KEY (was set — would conflict with local auth)"
  unset ANTHROPIC_API_KEY
fi

# 2. Verify oMLX is serving and the requested model is available.
echo "→ checking oMLX is serving..."
if ! models_json=$(curl -sf "$ENDPOINT/v1/models" -H "Authorization: Bearer $AUTH" 2>/dev/null); then
  echo ""
  echo "ERROR: oMLX is not responding at $ENDPOINT."
  echo "       Open oMLX and click 'Start Server' in the menu bar, then retry."
  exit 1
fi

if ! echo "$models_json" | grep -q "\"$MODEL\""; then
  echo ""
  echo "ERROR: model '$MODEL' is not in oMLX's served list."
  echo "       Available models:"
  echo "$models_json" | python3 -c 'import json,sys; [print("         -",m["id"]) for m in json.load(sys.stdin)["data"]]' 2>/dev/null \
    || echo "         (could not parse model list)"
  echo "       Pass a valid name:  ./launch_claude.sh MODEL_NAME"
  exit 1
fi

echo "→ oMLX up, model available."
echo "→ launching Claude Code (first request pays cold model-load, ~30-60s)..."
echo ""

# 3. Launch. All three tiers point at the same local model — Claude Code maps
#    its Opus/Sonnet/Haiku requests onto whatever we serve.
ANTHROPIC_BASE_URL="$ENDPOINT" \
ANTHROPIC_AUTH_TOKEN="$AUTH" \
ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" \
ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" \
ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
API_TIMEOUT_MS=3000000 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude
