#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
if [ -x env/bin/python ] && [ -f .runtime-location ] && [ "$(cat .runtime-location)" = "$SCRIPT_DIR" ]; then
    exit 0
fi
if ! mkdir .runtime-install-lock 2>/dev/null; then
    echo 'Runtime setup is already running. If a previous setup was interrupted, remove .runtime-install-lock and retry.' >&2
    exit 2
fi
trap 'rmdir .runtime-install-lock' EXIT
rm -f .runtime-location
rm -rf env
./offline/pixi-unpack ./offline/base-runtime.tar --output-directory "$SCRIPT_DIR" --shell bash
./env/bin/python -I -c 'import f8studio_server'
printf '%s\n' "$SCRIPT_DIR" > .runtime-location
