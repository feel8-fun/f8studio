#!/bin/sh

set -eu

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$repo_root"

mkdir -p packages/f8pysdk/f8pysdk/generated
pixi run -e default python -m datamodel_code_generator --input schemas/protocol.yml --input-file-type openapi --output-model-type pydantic_v2.BaseModel --output packages/f8pysdk/f8pysdk/generated/__init__.py --use-default --strict-nullable --allow-population-by-field-name --use-title-as-name --use-annotated
