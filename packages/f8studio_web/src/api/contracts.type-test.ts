// Compile-only contract tests. @ts-expect-error must continue to reject invalid wire values.
import type * as Wire from './contracts.gen';

const defaultInput: Wire.CreateProjectRequestInput = {};
const completeOutput: Wire.CreateProjectRequest = { name: 'Demo', description: '', projectId: null };
// @ts-expect-error Serialization emits default fields; consumers may rely on them.
const missingOutput: Wire.CreateProjectRequest = { name: 'Demo' };
const recursiveSchema: Wire.F8ArrayTypeSchema = { type: 'array', items: { type: 'array', items: { type: 'string' } } };
// @ts-expect-error Array schemas always require an item schema.
const missingItems: Wire.F8ArrayTypeSchema = { type: 'array' };
// @ts-expect-error Nullable is not equivalent to UNSET/omitted.
const invalidTitle: Wire.F8ArrayTypeSchema = { type: 'array', items: { type: 'string' }, title: null };
const request: Wire.ApiRequests['POST /api/projects'] = defaultInput;
// @ts-expect-error Request bodies are tied to their route.
const wrongRoute: Wire.ApiRequests['POST /api/projects/{project_id}/patch'] = defaultInput;
// @ts-expect-error Graph operation tags are a closed discriminated union.
const badOperation: Wire.PatchRequestInput['operations'][number] = { op: 'invented.operation' };
const position: Wire.SkeletonNode['pos'] = [1, 2, 3];
// @ts-expect-error A 3D position contains exactly three coordinates.
const invalidPosition: Wire.SkeletonNode['pos'] = [1, 2];
void [completeOutput, missingOutput, recursiveSchema, missingItems, invalidTitle, request, wrongRoute, badOperation, position, invalidPosition];
