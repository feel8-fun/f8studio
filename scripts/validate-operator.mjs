#!/usr/bin/env node
// Validate an operator catalog (array of operator specs) against schemas/protocol.yml (OpenAPI components.schemas) using Ajv.

import { readFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import Ajv from "ajv";
import YAML from "yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const protocolPath = path.join(root, "schemas", "protocol.yml");
const protocol = YAML.parse(readFileSync(protocolPath, "utf-8"));
const openapiSchemas = protocol?.components?.schemas;
if (!openapiSchemas || typeof openapiSchemas !== "object") {
  console.error(`Invalid OpenAPI spec (missing components.schemas): ${protocolPath}`);
  process.exit(1);
}

function rewriteOpenApiRefsToDefinitions(schema) {
  if (Array.isArray(schema)) return schema.map(rewriteOpenApiRefsToDefinitions);
  if (!schema || typeof schema !== "object") return schema;
  const out = {};
  for (const [k, v] of Object.entries(schema)) out[k] = rewriteOpenApiRefsToDefinitions(v);
  if (typeof out.$ref === "string" && out.$ref.startsWith("#/components/schemas/")) {
    out.$ref = out.$ref.replace("#/components/schemas/", "#/definitions/");
  }
  return out;
}

function buildJsonSchema(componentName) {
  const definitions = rewriteOpenApiRefsToDefinitions(structuredClone(openapiSchemas));
  return {
    $schema: "http://json-schema.org/draft-07/schema#",
    $id: `f8://schemas/${componentName}`,
    definitions,
    $ref: `#/definitions/${componentName}`,
  };
}

const fileArg = process.argv[2] || path.join(root, "services", "player", "operators.json");
const data = JSON.parse(readFileSync(fileArg, "utf-8"));

const ajv = new Ajv({ allErrors: true, strict: false });
const validate = ajv.compile(buildJsonSchema("F8OperatorSpec"));

const validateOne = (obj) => validate(obj);

let failed = false;
if (Array.isArray(data)) {
  data.forEach((op, idx) => {
    const ok = validateOne(op);
    if (!ok) {
      failed = true;
      console.error(`Invalid operator at index ${idx}:`, validate.errors);
    }
  });
} else {
  const ok = validateOne(data);
  if (!ok) {
    failed = true;
    console.error(`Invalid operator object:`, validate.errors);
  }
}

if (failed) {
  process.exit(1);
}

console.log(`Valid: ${fileArg}`);
