#!/usr/bin/env node
// Validate an operator catalog (array of operator specs) against schemas/operator.schema.json using Ajv.

import { readFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import Ajv from "ajv";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const schemaPath = path.join(root, "schemas", "operator.schema.json");
const commonPath = path.join(root, "schemas", "common.schema.json");
const schema = JSON.parse(readFileSync(schemaPath, "utf-8"));
const common = JSON.parse(readFileSync(commonPath, "utf-8"));

const fileArg = process.argv[2] || path.join(root, "services", "player", "operators.json");
const data = JSON.parse(readFileSync(fileArg, "utf-8"));

const ajv = new Ajv({ allErrors: true, strict: false });
ajv.addSchema(common, common.$id || "common.schema.json");
const validate = ajv.compile(schema);

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
