datamodel-codegen --input schemas/service.schema.json --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output packages/pyengine/f8engine/generated/service_spec.py

datamodel-codegen --input schemas/operator.schema.json --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output packages/pyengine/f8engine/generated/operator_spec.py