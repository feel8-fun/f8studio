#!/bin/sh

set -eu

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$repo_root"

export UV_CACHE_DIR="$repo_root/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

mkdir -p packages/pysdk/f8pysdk/generated
uv run python -m datamodel_code_generator --input schemas/protocol.yml --input-file-type openapi --output-model-type pydantic_v2.BaseModel --output packages/pysdk/f8pysdk/generated/__init__.py --use-default --strict-nullable --allow-population-by-field-name --use-title-as-name
