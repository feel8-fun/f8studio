#!/usr/bin/env node
// Validate a graph edge list/object against schemas/protocol.yml (OpenAPI components.schemas) using Ajv.

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

const fileArg = process.argv[2];
if (!fileArg) {
  console.error("Usage: validate-edge <file>");
  process.exit(1);
}

const data = JSON.parse(readFileSync(fileArg, "utf-8"));

const ajv = new Ajv({ allErrors: true, strict: false });
const validate = ajv.compile(buildJsonSchema("F8Edge"));

let failed = false;
if (Array.isArray(data)) {
  data.forEach((edge, idx) => {
    const ok = validate(edge);
    if (!ok) {
      failed = true;
      console.error(`Invalid edge at index ${idx}:`, validate.errors);
    }
  });
} else {
  const ok = validate(data);
  if (!ok) {
    failed = true;
    console.error("Invalid edge object:", validate.errors);
  }
}

if (failed) {
  process.exit(1);
}

console.log(`Valid: ${fileArg}`);
