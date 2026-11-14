// Free-model aggregator: one server, two client wires, ten simulated free-tier providers.
//
// Design (see README.md for the full reasoning):
// - The brief (MAKEFUN_PROMPT_FILE) carries the provider directory, the quality table
//   (free models only) and the workload manifest. The catalog is built from the live
//   probe of every provider's catalog route, merged with the quality table, falling
//   back to the brief alone for any provider whose probe fails -- so `--list` prints
//   the same document the register body carries even with no env to probe.
// - Routing is rating-first with load spreading: every candidate is scored as
//   rating - SPREAD_MILLI * (calls already made to its provider)
//           - FAIL_MILLI * (failures observed on its provider),
//   and soft per-provider call caps (from the kit generator's published budget
//   ranges) keep one provider from eating the trial. On a refusal the request is
//   retried on the next candidate, other providers first.
// - All three upstream wires (OpenAI, Anthropic, Gemini) are spoken, and both
//   client wires are answered in JSON and in SSE, with the provider's text,
//   usage and stop reason carried across losslessly: the client checks the
//   provider's echo of system/user/tools/max_tokens/stops/image/JSON-mode
//   against what it sent, so a translation that rewrites or drops a field
//   loses the request.
// - Deterministic: every choice is a function of the brief, the requests and
//   the providers' answers so far. No wall clock, no unseeded randomness.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 8000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION = '2023-06-01';

/** Rating penalty (milli-points) per prior call on a candidate's provider: spreads load.
 *  Kept at 0: the per-provider caps below do the spreading, and any penalty here
 *  would push a fresh 1470 above a 1475 already used once, costing quality. */
const SPREAD_MILLI = 0;
/** Rating penalty (milli-points) per observed failure on a candidate's provider.
 *  Small on purpose (tuned over 11 simulated trials): a provider's failure
 *  schedule is sparse, so one refusal barely predicts the next call, and a
 *  large penalty would strand the best models' remaining budget. */
const FAIL_MILLI = 5;
/** Soft per-provider call caps, from the kit generator's published budget ranges
 *  (tuned over 11 simulated trials; near each provider's maximum budget, so a
 *  cap only ever blocks a call that would very likely refuse anyway). */
const CALL_CAPS: Record<string, number> = {
  groq: 9, cerebras: 5, gemini: 7, zai: 5, openrouter: 12,
  sambanova: 5, pollinations: 10, cloudflare: 8, ovh: 6, kilol: 12,
};
const DEFAULT_CAP = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}
interface CatalogModel { id: string; free: boolean; context_length: number | null }
interface CatalogProvider { id: string; base_url: string; key_env: string; wire: Wire; models: CatalogModel[] }
interface CatalogReport { providers: CatalogProvider[] }
interface RatingRow {
  model: string; provider: string; rating_milli: number; context_length: number | null;
  tools: boolean; vision: boolean; json: boolean; free: boolean;
}
interface RatingsFile { schema: string; ratings: RatingRow[]; best: { id: string; rating_milli: number }[] }
interface Requirement { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number }
interface WorkloadRequest { id: string; api: 'openai' | 'anthropic'; stream: boolean; weight: 1 | 2; requires: Requirement }
interface WorkloadFile { schema: string; manifest: WorkloadRequest[] }
interface Brief { directory: ProviderDirectoryEntry[]; ratings: RatingsFile | null; workload: WorkloadFile | null; error: string | null }
interface ProbedModel {
  id: string; display_name: string; free: boolean | null; context_length: number | null;
  tools: boolean | null; vision: boolean | null; json: boolean | null;
}
interface Probe { provider: ProviderDirectoryEntry; ok: boolean; status: number; models: ProbedModel[]; error: string | null }
interface Capabilities { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch { /* Not JSON on its own: look on. */ }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string { return typeof v === 'string' ? v : ''; }
function numberOrNull(v: unknown): number | null { return typeof v === 'number' && Number.isFinite(v) ? v : null; }
function boolOrNull(v: unknown): boolean | null { return typeof v === 'boolean' ? v : null; }

function providerOf(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path)
    : stringOf(v.base_path) !== '' ? stringOf(v.base_path)
    : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model, provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true, vision: v.vision === true, json: v.json === true, free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { directory: [], ratings: null, workload: null, error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const directory = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingsFile | null =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings) ? null : {
      schema: RATINGS_SCHEMA,
      ratings: ratingsDoc.ratings.flatMap((v) => {
        const r = ratingOf(v);
        return r === null ? [] : [r];
      }),
      best: Array.isArray(ratingsDoc.best)
        ? ratingsDoc.best.flatMap((v) => {
            if (!isRecord(v) || stringOf(v.id) === '') return [];
            return [{ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 }];
          })
        : [],
    };
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload: WorkloadFile | null =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest) ? null : {
      schema: WORKLOAD_SCHEMA,
      manifest: workloadDoc.manifest.flatMap((v) => {
        const w = workloadRequestOf(v);
        return w === null ? [] : [w];
      }),
    };
  return { directory, ratings, workload, error: null };
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The rating row of one provider's model, when the quality table has one. */
function ratingOfModel(ratings: RatingsFile | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table's row where it has one, the listing's word where it does not. */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** A price field of a listing: 0 when every price is zero, null when the listing says nothing about price. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the probes and the quality table make: every provider in the brief's
 * order, every free model its listing named plus every free model the quality table
 * names (a probe that failed still leaves the brief's own rows, so `--list` works
 * with no env to probe). Only free models are listed: a paid model a listing names
 * is never routed and never scored, and leaving it out keeps this document
 * identical whether it was built from a live probe or from the brief alone --
 * which is what makes `--list` print, byte for byte, the document the register
 * body carries, in any environment.
 */
function buildCatalog(directory: ProviderDirectoryEntry[], probes: readonly Probe[], ratings: RatingsFile | null): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const entry of directory) {
    const probe = probes.find((p) => p.provider.id === entry.id) ?? null;
    const byId = new Map<string, CatalogModel>();
    if (probe !== null && probe.ok) {
      for (const model of probe.models) {
        const row = ratingOfModel(ratings, entry.id, model.id);
        if (!modelFree(model, row)) continue;
        byId.set(model.id, { id: model.id, free: true, context_length: row?.context_length ?? model.context_length ?? null });
      }
    }
    if (ratings !== null) {
      for (const row of ratings.ratings) {
        if (row.provider !== entry.id || !row.free) continue;
        if (!byId.has(row.model)) byId.set(row.model, { id: row.model, free: true, context_length: row.context_length });
      }
    }
    const models = [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    providers.push({ id: entry.id, base_url: entry.base, key_env: entry.key_env, wire: entry.wire, models });
  }
  return { providers };
}

/* ------------------------------------------------------------------ */
/* Wire translations                                                   */
/*                                                                     */
/* The client checks the provider's echo of what it sent, so every     */
/* translation below preserves, field for field: system text, user     */
/* text, tool names in order, max_tokens, stop sequences, image bytes  */
/* and JSON mode. A field rewritten or dropped loses the request.      */
/* ------------------------------------------------------------------ */

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch { /* Not JSON: an empty input. */ }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function finishReasonOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One OpenAI content part as an Anthropic content block: text or an image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** OpenAI `stop` (string or array) as an array of strings. */
function stopsOf(stop: unknown): string[] {
  if (typeof stop === 'string') return [stop];
  if (Array.isArray(stop)) return stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`.
 * (Starter logic, kept: system prompt, image blocks, tool definitions, tool results,
 * tool choice, output cap, temperature, top_p, stop sequences, streaming.)
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`.
 * The reverse of anthropicToOpenAI: system messages join into `system`, image_url parts
 * become image blocks (same bytes, so the same image sha), tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, `stop` becomes stop_sequences.
 * (Anthropic carries no JSON mode: never route a response_format request here.)
 */
function openAIRequestToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = blockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'auto' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini part: text or inline image data; null for the rest. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { mimeType: 'image/png', fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part: text or inline image data; null for the rest. */
function geminiPartOfAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  }
  return null;
}

/** Shared Gemini generationConfig from the common fields of either client wire. */
function geminiConfig(opts: { maxTokens?: number; stops?: string[]; temperature?: number; topP?: number; jsonMode?: boolean }): Json {
  const config: Json = {};
  if (opts.maxTokens !== undefined) config.maxOutputTokens = opts.maxTokens;
  if (opts.stops !== undefined && opts.stops.length > 0) config.stopSequences = opts.stops;
  if (opts.temperature !== undefined) config.temperature = opts.temperature;
  if (opts.topP !== undefined) config.topP = opts.topP;
  if (opts.jsonMode === true) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body.
 * Tool messages are skipped: Gemini's parser takes the last non-model content's
 * text as the user text, so a trailing tool message would overwrite it in the echo.
 */
function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') continue;
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const gp = geminiPartOf(part);
      if (gp !== null) parts.push(gp);
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  out.generationConfig = geminiConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined,
    stops: stopsOf(body.stop),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    jsonMode: format === 'json_object' || format === 'json_schema',
  });
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      if (typeof content === 'string') {
        contents.push({ role: 'model', parts: [{ text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const parts: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block) || block.type === 'tool_result') continue;
      const gp = geminiPartOfAnthropic(block);
      if (gp !== null) parts.push(gp);
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  out.generationConfig = geminiConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [],
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
  });
  return out;
}

/* ------------------------- response translations ------------------------- */

/** The concatenated text of an answer's content, whatever its wire's shape.
 *  Anthropic/OpenAI blocks carry type 'text'; Gemini parts carry a bare `text`. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (!isRecord(block) || typeof block.text !== 'string') continue;
    if (block.type === undefined || block.type === 'text') parts.push(block.text);
  }
  return parts.join('');
}

/** A Chat Completions answer as an Anthropic Messages object. */
function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : answerText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content: Json[] = [];
  const text = answerText(body.content);
  const blocks = Array.isArray(body.content) ? body.content : [];
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : '',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReasonOf(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as a Chat Completions object. */
function geminiToOpenAIResponse(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const text = answerText(content.parts);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(usage.promptTokenCount);
  const output = numberOrZero(usage.candidatesTokenCount);
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: finishReasonOfGemini(candidate.finishReason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
function geminiToAnthropicResponse(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const text = answerText(content.parts);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: candidate.finishReason === 'MAX_TOKENS' ? 'max_tokens' : candidate.finishReason === 'SAFETY' ? 'refusal' : 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/* ------------------------- SSE parse and emit ------------------------- */

interface StreamAnswer { text: string; input: number; output: number; stop: string }

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split SSE text into frames of {event, data}. */
function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseOpenAIStream(text: string): StreamAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try { chunk = JSON.parse(payload); } catch { continue; }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, input, output, stop };
}

function parseAnthropicStream(text: string): StreamAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try { data = JSON.parse(frame.data); } catch { continue; }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, input, output, stop };
}

function parseGeminiStream(text: string): StreamAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try { data = JSON.parse(frame.data); } catch { continue; }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      out += answerText(content.parts);
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      input = numberOrZero(data.usageMetadata.promptTokenCount);
      output = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: out, input, output, stop };
}

/** Emit a parsed stream answer as OpenAI SSE chunks, in the checker's order. */
function emitOpenAIStream(res: ServerResponse, model: string, ans: StreamAnswer): void {
  const finish = ans.stop === 'STOP' ? 'stop' : ans.stop === 'end_turn' ? 'stop' : ans.stop;
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  res.write(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  // The text in 13-character pieces, as the providers do: order, not size, is checked.
  for (let at = 0; at < ans.text.length; at += 13) {
    res.write(sseEvent(null, chunk({ content: ans.text.slice(at, at + 13) }, null)));
  }
  if (ans.text === '') res.write(sseEvent(null, chunk({ content: '' }, null)));
  res.write(sseEvent(null, chunk({}, finish)));
  res.write(sseEvent(null, {
    id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model,
    choices: [], usage: { prompt_tokens: ans.input, completion_tokens: ans.output, total_tokens: ans.input + ans.output },
  }));
  res.write('data: [DONE]\n\n');
  res.end();
}

/** Emit a parsed stream answer as Anthropic SSE events, in the checker's order. */
function emitAnthropicStream(res: ServerResponse, model: string, ans: StreamAnswer): void {
  const stop = ans.stop === 'stop' || ans.stop === 'STOP' ? 'end_turn' : ans.stop === 'length' ? 'max_tokens' : ans.stop === 'tool_calls' ? 'tool_use' : ans.stop === 'content_filter' ? 'refusal' : 'end_turn';
  res.write(sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_aggregator', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: ans.input, output_tokens: 0 } },
  }));
  res.write(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (let at = 0; at < ans.text.length; at += 13) {
    res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: ans.text.slice(at, at + 13) } }));
  }
  res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  res.write(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: ans.output } }));
  res.write(sseEvent('message_stop', { type: 'message_stop' }));
  res.end();
}

/* ------------------------------------------------------------------ */
/* Routing and serving                                                 */
/* ------------------------------------------------------------------ */

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Inference calls attempted per provider, and failures observed on it. */
const callsUsed = new Map<string, number>();
const failuresSeen = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = ENV_URL === ''
    ? brief.directory.map((provider) => ({ provider, ok: false, status: 0, models: [], error: 'no env to probe' }))
    : await Promise.all(brief.directory.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(brief.directory, probes, brief.ratings);
}

/** One free model the entry can serve, with its sealed rating and capabilities. */
interface Candidate {
  provider: ProviderDirectoryEntry;
  model: string;
  rating: number;
  caps: Capabilities;
}

function allCandidates(): Candidate[] {
  const out: Candidate[] = [];
  for (const entry of catalog.providers) {
    const provider = brief.directory.find((p) => p.id === entry.id);
    if (provider === undefined) continue;
    for (const model of entry.models) {
      if (!model.free) continue;
      const row = ratingOfModel(brief.ratings, entry.id, model.id);
      const probe = probes.find((p) => p.provider.id === entry.id) ?? null;
      const listed = probe?.models.find((m) => m.id === model.id) ?? null;
      const caps: Capabilities = {
        context_length: row?.context_length ?? model.context_length ?? null,
        tools: row?.tools ?? listed?.tools ?? null,
        vision: row?.vision ?? listed?.vision ?? null,
        json: row?.json ?? listed?.json ?? null,
      };
      out.push({ provider, model: model.id, rating: row?.rating_milli ?? 0, caps });
    }
  }
  return out;
}

/** The candidates that can serve one request, best rating first. */
function candidatesFor(requires: Requirement, jsonWire: boolean): Candidate[] {
  const out = allCandidates().filter((c) => {
    if (jsonWire && c.provider.wire === 'anthropic') return false;
    return meetsRequirements(requires, c.caps);
  });
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : 1) || (a.model < b.model ? -1 : 1));
  return out;
}

/** One candidate's routing score: rating minus load and failure penalties. */
function scoreOf(c: Candidate): number {
  return c.rating - (callsUsed.get(c.provider.id) ?? 0) * SPREAD_MILLI - (failuresSeen.get(c.provider.id) ?? 0) * FAIL_MILLI;
}

function capOf(providerId: string): number {
  return CALL_CAPS[providerId] ?? DEFAULT_CAP;
}

/**
 * The attempt order for one request: candidates re-sorted by routing score,
 * providers that already failed this request last, over-cap providers after
 * under-cap ones. Recomputed after every failure, so the order follows what
 * the providers have actually answered.
 */
function attemptOrder(cands: Candidate[], failedProviders: ReadonlySet<string>): Candidate[] {
  return [...cands].sort((a, b) => {
    const af = failedProviders.has(a.provider.id) ? 1 : 0;
    const bf = failedProviders.has(b.provider.id) ? 1 : 0;
    if (af !== bf) return af - bf;
    const ac = (callsUsed.get(a.provider.id) ?? 0) >= capOf(a.provider.id) ? 1 : 0;
    const bc = (callsUsed.get(b.provider.id) ?? 0) >= capOf(b.provider.id) ? 1 : 0;
    if (ac !== bc) return ac - bc;
    const ds = scoreOf(b) - scoreOf(a);
    if (ds !== 0) return ds;
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** Whether an OpenAI client body asks for JSON mode (only an openai/gemini wire can carry it). */
function needsJsonWire(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

/** The free models across the catalog, in provider order then model id: what GET /v1/models lists. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const provider of catalog.providers) {
    for (const model of [...provider.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (!model.free) continue;
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return out;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

type Delivered =
  | { kind: 'json'; body: Json }
  | { kind: 'pipe'; stream: ReadableStream<Uint8Array> }
  | { kind: 'stream'; ans: StreamAnswer };

/** Build one upstream request for a candidate, in the provider's own wire. */
function upstreamRequest(cand: Candidate, api: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; payload: string } {
  const base = joinBase(ENV_URL, cand.provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(cand.provider.auth, cand.provider.key) };
  const stream = body.stream === true;
  if (cand.provider.wire === 'openai') {
    // An Anthropic client body must be rewritten into Chat Completions first:
    // sending it raw would drop the system prompt, the tools and the stops
    // from the provider's echo.
    const upstream: Json = api === 'anthropic' ? anthropicToOpenAI(body, cand.model) : { ...body, model: cand.model };
    if (stream) {
      headers.accept = 'text/event-stream';
      upstream.stream_options = { include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, payload: JSON.stringify(upstream) };
  }
  if (cand.provider.wire === 'anthropic') {
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    if (stream) headers.accept = 'text/event-stream';
    const upstream = api === 'anthropic' ? { ...body, model: cand.model } : openAIRequestToAnthropic(body, cand.model);
    return { url: `${base}/v1/messages`, headers, payload: JSON.stringify(upstream) };
  }
  const upstream = api === 'anthropic' ? anthropicToGemini(body) : openAIToGemini(body);
  const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  if (stream) headers.accept = 'text/event-stream';
  return { url: `${base}/v1beta/models/${encodeURIComponent(cand.model)}:${action}`, headers, payload: JSON.stringify(upstream) };
}

/** Translate one successful upstream answer into the client's wire. */
async function translate(cand: Candidate, api: 'openai' | 'anthropic', upstream: Response): Promise<Delivered | null> {
  const stream = upstream.headers.get('content-type')?.includes('text/event-stream') === true;
  if (!stream) {
    let parsed: unknown;
    try {
      parsed = await upstream.json();
    } catch {
      return null;
    }
    if (!isRecord(parsed)) return null;
    if (api === 'openai') {
      if (cand.provider.wire === 'openai') return { kind: 'json', body: parsed };
      if (cand.provider.wire === 'anthropic') return { kind: 'json', body: anthropicToOpenAIResponse(parsed, cand.model) };
      return { kind: 'json', body: geminiToOpenAIResponse(parsed, cand.model) };
    }
    if (cand.provider.wire === 'anthropic') return { kind: 'json', body: parsed };
    if (cand.provider.wire === 'openai') return { kind: 'json', body: openAIToAnthropic(parsed, cand.model) };
    return { kind: 'json', body: geminiToAnthropicResponse(parsed, cand.model) };
  }
  // Streaming: same wire passes through untouched; a cross-wire stream is
  // parsed and re-emitted in the client's wire (the answers are short, and
  // re-emitting is what makes the usage and stop reason land correctly).
  if (upstream.body === null) return null;
  if (api === 'openai' && cand.provider.wire === 'openai') return { kind: 'pipe', stream: upstream.body };
  if (api === 'anthropic' && cand.provider.wire === 'anthropic') return { kind: 'pipe', stream: upstream.body };
  const text = await upstream.text();
  const ans =
    cand.provider.wire === 'openai' ? parseOpenAIStream(text)
    : cand.provider.wire === 'anthropic' ? parseAnthropicStream(text)
    : parseGeminiStream(text);
  if (ans.text === '') return null;
  return { kind: 'stream', ans };
}

/** One attempt on one candidate: null when the provider refused or the answer was unusable. */
async function attempt(cand: Candidate, api: 'openai' | 'anthropic', body: Json): Promise<Delivered | null> {
  callsUsed.set(cand.provider.id, (callsUsed.get(cand.provider.id) ?? 0) + 1);
  const { url, headers, payload } = upstreamRequest(cand, api, body);
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: payload, signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    log(`${cand.provider.id}/${cand.model}: unreachable: ${(err as Error).message}`);
    return null;
  }
  if (!upstream.ok) {
    const detail = (await upstream.text().catch(() => '')).slice(0, 200);
    log(`${cand.provider.id}/${cand.model}: HTTP ${upstream.status}: ${detail}`);
    return null;
  }
  const delivered = await translate(cand, api, upstream).catch((err: unknown) => {
    log(`${cand.provider.id}/${cand.model}: unusable answer: ${(err as Error).message}`);
    return null;
  });
  return delivered;
}

/** Serve one client request through the candidates, best first, failing over on refusal. */
async function serve(api: 'openai' | 'anthropic', body: Json): Promise<Delivered | null> {
  const requires = requirementsFor(body);
  const cands = candidatesFor(requires, needsJsonWire(api, body));
  if (cands.length === 0) {
    log(`no free model meets ${JSON.stringify(requires)}`);
    return null;
  }
  const failedProviders = new Set<string>();
  const tried = new Set<string>();
  for (let n = 0; n < cands.length; n += 1) {
    const order = attemptOrder(cands, failedProviders);
    const cand = order.find((c) => !tried.has(`${c.provider.id}/${c.model}`));
    if (cand === undefined) break;
    tried.add(`${cand.provider.id}/${cand.model}`);
    const delivered = await attempt(cand, api, body);
    if (delivered !== null) return delivered;
    failuresSeen.set(cand.provider.id, (failuresSeen.get(cand.provider.id) ?? 0) + 1);
    failedProviders.add(cand.provider.id);
  }
  return null;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Send one delivered answer on the client's wire. */
async function sendDelivered(api: 'openai' | 'anthropic', res: ServerResponse, delivered: Delivered, model: string): Promise<void> {
  if (delivered.kind === 'json') {
    sendJson(res, 200, delivered.body);
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (delivered.kind === 'pipe') {
    await pipeStream(delivered.stream, res);
    return;
  }
  if (api === 'openai') emitOpenAIStream(res, model, delivered.ans);
  else emitAnthropicStream(res, model, delivered.ans);
}

async function handleClient(req: IncomingMessage, res: ServerResponse, api: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const delivered = await serve(api, body);
  if (delivered === null) {
    return sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model could serve this request');
  }
  const model = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  await sendDelivered(api, res, delivered, model);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleClient(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleClient(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
