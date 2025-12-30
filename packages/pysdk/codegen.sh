#!/bin/sh

cd $(dirname "$0")
mkdir -p f8pysdk/generated
datamodel-codegen --input ../../schemas/specs.schema.json --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output f8pysdk/generated/ --use-default --strict-nullable --allow-population-by-field-name --use-title-as-name --class-name F8Specs