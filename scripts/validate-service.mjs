#!/usr/bin/env node
// Validate a service profile against schemas/service.schema.json using Ajv.

import { readFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import Ajv from "ajv";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const schemaPath = path.join(root, "schemas", "service.schema.json");
const commonPath = path.join(root, "schemas", "common.schema.json");
const schema = JSON.parse(readFileSync(schemaPath, "utf-8"));
const common = JSON.parse(readFileSync(commonPath, "utf-8"));

const fileArg = process.argv[2] || path.join(root, "services", "player", "service.json");
const data = JSON.parse(readFileSync(fileArg, "utf-8"));

const ajv = new Ajv({ allErrors: true, strict: false });
ajv.addSchema(common, common.$id || "common.schema.json");
const validate = ajv.compile(schema);
const ok = validate(data);

if (!ok) {
  console.error(`Invalid: ${fileArg}`);
  console.error(validate.errors);
  process.exit(1);
}

console.log(`Valid: ${fileArg}`);
