// Thin TS client adapter for NATS request/reply using generated models.
// Assumes you generated types into packages/shared/generated/master.

import { StringCodec, NatsConnection } from "nats";
import { PingRequest, PingReply, ApplyConfigRequest, ApplyConfigReply } from "./generated/master";

const sc = StringCodec();

function encode(body: unknown): Uint8Array {
  return sc.encode(JSON.stringify(body));
}

function decode<T>(data: Uint8Array): T {
  return JSON.parse(sc.decode(data)) as T;
}

// Reuse the same msgId on retries for idempotency/dedup.
export async function ping(
  nc: NatsConnection,
  body: PingRequest,
  timeoutMs = 1000
): Promise<PingReply> {
  const msg = await nc.request("f8.master.ping", encode(body), { timeout: timeoutMs });
  return decode<PingReply>(msg.data);
}

export async function applyConfig(
  nc: NatsConnection,
  body: ApplyConfigRequest,
  timeoutMs = 3000
): Promise<ApplyConfigReply> {
  const msg = await nc.request("f8.master.config.apply", encode(body), { timeout: timeoutMs });
  return decode<ApplyConfigReply>(msg.data);
}

// Pattern for other RPCs: map operationId -> x-nats-subject, set timeout, decode reply.
