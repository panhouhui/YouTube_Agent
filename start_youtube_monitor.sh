#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/log"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HTTP_PROXY="http://127.0.0.1:7897"
export HTTPS_PROXY="http://127.0.0.1:7897"
export ALL_PROXY="http://127.0.0.1:7897"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export all_proxy="$ALL_PROXY"
exec "$ROOT/.venv/bin/python" -m avtdl.avtdl --config config.youtube.rootlog.yml "$@"
