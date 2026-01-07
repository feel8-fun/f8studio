#!/bin/sh

cd $(dirname "$0")
mkdir -p f8pysdk/generated
python -m datamodel_code_generator --input ../../schemas/specs.schema.json --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output f8pysdk/generated/__init__.py --use-default --strict-nullable --allow-population-by-field-name --use-title-as-name --class-name F8Specs
