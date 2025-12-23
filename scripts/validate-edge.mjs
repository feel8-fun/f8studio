#!/usr/bin/env node
// Validate a graph edge list/object against schemas/edge.schema.json using Ajv.

import { readFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import Ajv from "ajv";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const schemaPath = path.join(root, "schemas", "edge.schema.json");
const schema = JSON.parse(readFileSync(schemaPath, "utf-8"));

const fileArg = process.argv[2];
if (!fileArg) {
  console.error("Usage: validate-edge <file>");
  process.exit(1);
}

const data = JSON.parse(readFileSync(fileArg, "utf-8"));

const ajv = new Ajv({ allErrors: true, strict: false });
const validate = ajv.compile(schema);

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
