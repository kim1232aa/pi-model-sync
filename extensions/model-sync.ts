import { execFile } from "node:child_process";
import { readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";

/**
 * Generic model catalog sync for pi / pi-web.
 *
 * For every provider in models.json that declares a `baseUrl`, this extension
 * fetches the provider's model list, derives each model's capabilities from the
 * upstream payload first (falling back to models.dev), and rewrites that
 * provider's `models` array in models.json.
 *
 * Nothing is hardcoded per provider: add a provider in pi (or the pi-web Models
 * panel), then run `/refresh-custom-models`.
 *
 * Capability sources, highest priority first:
 *   1. `<provider>-overrides.json` in the agent dir
 *   2. the upstream model list payload (many response shapes are understood)
 *   3. the models.dev catalog (matched by `provider/model` or bare model id)
 *   4. a small gpt-5.x heuristic and an id-based reasoning hint, then defaults
 *
 * Providers without `baseUrl` (for example a `compat`-only override of a
 * built-in provider) are never touched. Providers whose fetch fails keep their
 * existing models.
 */

const execFileAsync = promisify(execFile);

const AGENT_DIR = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
const MODELS_JSON_PATH = join(AGENT_DIR, "models.json");
const AUTH_PATH = join(AGENT_DIR, "auth.json");
const MODELS_DEV_URL = "https://models.dev/api.json";
const MODELS_DEV_TTL_MS = 6 * 60 * 60 * 1000;
const REQUEST_TIMEOUT_MS = 30_000;

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

/** Model ids that are clearly not chat/completions models. */
const NON_CHAT_ID = /(?:^openai-compatible-chat|(?:^|\/)no-think\/|embedding|rerank|transcri|\btts\b|\baudio\b|realtime|background-task|gpt-image|agnes-image|agnes-video|seedream|seedance|seededit|seed3d|hitem3d|hyper3d|grok-imagine|happyhorse|(?:^|[-_/])(?:i2v|t2v|flf2v|r2v)(?:[-_/]|$)|gemini-(?:image|imagen)|imagen-)/i;

/** thinkingFormat values pi understands (see docs/models.md). */
const PI_THINKING_FORMATS = new Set([
  "openai", "openrouter", "deepseek", "together", "baseten", "zai",
  "qwen", "chat-template", "qwen-chat-template", "string-thinking", "ant-ling",
]);

type AnyRec = Record<string, any>;

interface ModelOverride {
  contextWindow?: number;
  maxTokens?: number;
  reasoning?: boolean;
  vision?: boolean;
  thinkingLevelMap?: Record<string, string | null>;
  filter?: boolean;
}

interface Cost { input: number; output: number; cacheRead: number; cacheWrite: number }

interface SyncedModel {
  id: string;
  name: string;
  reasoning: boolean;
  input: string[];
  contextWindow: number;
  maxTokens: number;
  cost: Cost;
  thinkingLevelMap?: Record<string, string | null>;
  compat?: AnyRec;
}

// ---------------------------------------------------------------------------
// tiny value helpers
// ---------------------------------------------------------------------------

function isRec(value: unknown): value is AnyRec {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asBool(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function asNum(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asPos(value: unknown): number | undefined {
  const n = asNum(value);
  return n !== undefined && n > 0 ? n : undefined;
}

function asStr(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function pick<T>(...values: (T | undefined)[]): T | undefined {
  for (const value of values) if (value !== undefined) return value;
  return undefined;
}

function getPath(obj: unknown, path: string): unknown {
  let cur: unknown = obj;
  for (const key of path.split(".")) {
    if (!isRec(cur)) return undefined;
    cur = cur[key];
  }
  return cur;
}

function getAt(obj: unknown, ...keys: string[]): unknown {
  return keys.map((key) => getPath(obj, key)).find((value) => value !== undefined);
}

/** Returns true/false when an input-modality list is present and non-empty. */
function includesImage(...lists: unknown[]): boolean | undefined {
  for (const list of lists) {
    if (!Array.isArray(list)) continue;
    const items = list.filter((value): value is string => typeof value === "string");
    if (items.length > 0) return items.some((value) => /image|vision/i.test(value));
  }
  return undefined;
}

async function readJson<T>(path: string): Promise<T | undefined> {
  try {
    return JSON.parse(await readFile(path, "utf8")) as T;
  } catch {
    return undefined;
  }
}

/** pi accepts // and /* *\/ comments in models.json; strip them without touching strings (URLs contain //). */
export function stripJsonComments(text: string): string {
  let out = "";
  let inString = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const next = text[i + 1];
    if (inString) {
      out += c;
      if (c === "\\") out += text[++i] ?? "";
      else if (c === '"') inString = false;
    } else if (c === '"') {
      inString = true;
      out += c;
    } else if (c === "/" && next === "/") {
      while (i < text.length && text[i] !== "\n") i++;
      out += "\n";
    } else if (c === "/" && next === "*") {
      i += 2;
      while (i < text.length && !(text[i] === "*" && text[i + 1] === "/")) i++;
      i++;
    } else {
      out += c;
    }
  }
  return out;
}

/**
 * Read models.json. A missing file is an empty config; an unparseable file is an
 * error, so the caller never overwrites a config it could not read.
 */
export async function readModelsJson(path: string): Promise<{ providers?: Record<string, AnyRec> }> {
  let text: string;
  try {
    text = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return {};
    throw error;
  }
  const parsed: unknown = JSON.parse(stripJsonComments(text.replace(/^\uFEFF/, "")));
  if (!isRec(parsed)) throw new Error(`${path} is not a JSON object`);
  return parsed;
}

/** Write via temp file + rename so a crash can never leave a truncated models.json. */
async function writeFileAtomic(path: string, content: string): Promise<void> {
  const tmp = `${path}.${process.pid}.tmp`;
  await writeFile(tmp, content, { mode: 0o600 });
  await rename(tmp, path);
}

// ---------------------------------------------------------------------------
// config value resolution (same syntax as models.json: literal / $ENV / !cmd)
// ---------------------------------------------------------------------------

function interpolateEnv(input: string): string | undefined {
  let missing = false;
  const out = input.replace(
    /\$\$|\$!|\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)/g,
    (match, braced: string | undefined, bare: string | undefined) => {
      if (match === "$$") return "$";
      if (match === "$!") return "!";
      const name = braced ?? bare ?? "";
      const value = process.env[name];
      if (value === undefined) {
        missing = true;
        return "";
      }
      return value;
    },
  );
  return missing ? undefined : out;
}

async function resolveConfigValue(raw: string): Promise<string | undefined> {
  if (raw.startsWith("!") && !raw.startsWith("$!")) {
    try {
      const { stdout } = await execFileAsync("/bin/sh", ["-c", raw.slice(1)], { timeout: REQUEST_TIMEOUT_MS });
      return stdout.trim() || undefined;
    } catch {
      return undefined;
    }
  }
  const resolved = interpolateEnv(raw);
  return resolved && resolved.trim() ? resolved : undefined;
}

/**
 * Resolve a provider's API key the way pi does: the provider's own `apiKey`
 * (literal, `$ENV`, or `!command`), then pi's stored credentials (auth.json,
 * written by `/login`).
 */
async function resolveApiKey(provider: AnyRec, providerName: string, auth: AnyRec): Promise<string | undefined> {
  const inline = asStr(provider.apiKey);
  if (inline) {
    const resolved = await resolveConfigValue(inline);
    if (resolved) return resolved;
  }
  const stored = auth[providerName];
  if (isRec(stored)) {
    if (typeof stored.key === "string" && stored.key) return stored.key;
    if (typeof stored.access === "string" && stored.access) return stored.access;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// upstream endpoint & candidate discovery
// ---------------------------------------------------------------------------

export function getCandidateModelUrls(baseUrl: string, api: string = "openai-completions"): string[] {
  let clean = baseUrl.replace(/\/+$/, "");
  if (api === "anthropic-messages") {
    const base = /\/v\d+(?:beta)?$/i.test(clean) ? clean : `${clean}/v1`;
    const url = /\/models$/i.test(base) ? base : `${base}/models`;
    return [`${url}?limit=1000`];
  }
  if (api === "google-generative-ai") {
    if (/\/models$/i.test(clean)) return [clean];
    if (/\/v\d+(?:beta)?$/i.test(clean)) return [`${clean}/models`];
    return [`${clean}/v1beta/models`, `${clean}/models`];
  }

  // Generic OpenAI-compatible / default APIs
  if (/\/models$/i.test(clean)) return [clean];
  if (/\/v\d+(?:beta)?$/i.test(clean)) return [`${clean}/models`];

  // When baseUrl lacks a version suffix like /v1, candidate 1 is /v1/models, candidate 2 is /models
  return [`${clean}/v1/models`, `${clean}/models`];
}

async function requestHeaders(api: string, apiKey: string | undefined, extra: unknown): Promise<Record<string, string>> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (isRec(extra)) {
    // Header values use the same $ENV / !command syntax as apiKey.
    for (const [key, value] of Object.entries(extra)) {
      const resolved = typeof value === "string" ? await resolveConfigValue(value) : undefined;
      if (resolved) headers[key] = resolved;
    }
  }
  const hasAuth = Object.keys(headers).some((key) => /^(authorization|x-api-key|x-goog-api-key)$/i.test(key));
  if (apiKey && !hasAuth) {
    if (api === "anthropic-messages") {
      headers["x-api-key"] = apiKey;
      headers["anthropic-version"] ??= "2023-06-01";
    } else if (api === "google-generative-ai") {
      headers["x-goog-api-key"] = apiKey;
    } else {
      headers.Authorization = `Bearer ${apiKey}`;
    }
  }
  return headers;
}

async function fetchModelList(
  baseUrl: string,
  api: string,
  apiKey: string | undefined,
  extraHeaders: unknown,
  signal?: AbortSignal,
): Promise<AnyRec[]> {
  const candidates = getCandidateModelUrls(baseUrl, api);
  const headers = await requestHeaders(api, apiKey, extraHeaders);
  let lastError: Error | null = null;

  for (const url of candidates) {
    try {
      const response = await fetch(url, {
        headers,
        signal: signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      if (response.status === 404 || response.status === 405) {
        lastError = new Error(`${url} returned HTTP ${response.status}`);
        continue;
      }
      const text = await response.text();
      if (!response.ok) {
        throw new Error(`${url} returned HTTP ${response.status}${text ? `: ${text.slice(0, 200)}` : ""}`);
      }
      let payload: unknown;
      try {
        payload = JSON.parse(text);
      } catch {
        throw new Error(`${url} returned invalid JSON`);
      }
      const list = extractModelList(payload);
      if (list.length === 0) throw new Error(`${url} returned no models`);
      return list;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (signal?.aborted) throw err;
    }
  }

  throw lastError || new Error(`Failed to fetch models from candidate endpoints: ${candidates.join(", ")}`);
}

function extractModelList(payload: unknown): AnyRec[] {
  if (Array.isArray(payload)) return payload.filter(isRec);
  if (isRec(payload)) {
    for (const key of ["data", "models", "results", "items"]) {
      const value = payload[key];
      if (Array.isArray(value)) return value.filter(isRec);
      if (isRec(value)) return Object.values(value).filter(isRec);
    }
  }
  return [];
}

function normalizeEntry(raw: AnyRec): { id: string; name?: string } | undefined {
  const rawId = asStr(raw.id) ?? asStr(raw.model) ?? asStr(raw.name);
  if (!rawId) return undefined;
  const id = rawId.replace(/^models\//, "");
  if (!id) return undefined;
  const explicitId = asStr(raw.id) ?? asStr(raw.model);
  const display = asStr(raw.display_name) ?? asStr(raw.displayName) ?? (explicitId ? asStr(raw.name) : undefined);
  return { id, name: display && display !== id ? display : undefined };
}

// ---------------------------------------------------------------------------
// capability extraction from an upstream model entry
// ---------------------------------------------------------------------------

interface UpstreamCapabilities {
  reasoning?: boolean;
  vision?: boolean;
  contextWindow?: number;
  maxTokens?: number;
  thinkingFormat?: string;
  thinkingCanDisable?: boolean;
  thinkingEffortSupported?: boolean;
}

export function extractUpstream(raw: AnyRec): UpstreamCapabilities {
  const caps = isRec(raw.capabilities) ? raw.capabilities : undefined;
  const params = Array.isArray(raw.supported_parameters)
    ? raw.supported_parameters.filter((value): value is string => typeof value === "string")
    : [];

  const reasoning = pick(
    asBool(caps?.reasoning),
    asBool(caps?.supports_reasoning),
    asBool(raw.reasoning),
    asBool(raw.supports_reasoning),
    caps?.supportsThinking === true || caps?.thinking === true ? true : undefined,
    params.some((p) => /reason|thinking/i.test(p)) ? true : undefined,
  );

  const vision = pick(
    asBool(caps?.vision),
    asBool(caps?.supports_vision),
    asBool(raw.supports_vision),
    asBool(raw.vision),
    includesImage(
      caps?.supported_modalities,
      caps?.input_modalities,
      getAt(raw, "architecture.input_modalities"),
      raw.input_modalities,
      raw.supported_modalities,
      getAt(raw, "modalities.input"),
    ),
  );

  const contextWindow = pick(
    asPos(caps?.contextWindow),
    asPos(caps?.context_window),
    asPos(raw.contextWindow),
    asPos(raw.context_window),
    asPos(raw.context_length),
    asPos(raw.max_input_tokens),
    asPos(raw.max_context_length),
    asPos(raw.inputTokenLimit),
    asPos(getAt(raw, "limit.context")),
    asPos(getAt(raw, "top_provider.context_length")),
  );

  const maxTokens = pick(
    asPos(caps?.maxOutput),
    asPos(caps?.maxOutputTokens),
    asPos(caps?.max_output_tokens),
    asPos(raw.max_output_tokens),
    asPos(raw.max_tokens),
    asPos(raw.outputTokenLimit),
    asPos(raw.max_completion_tokens),
    asPos(getAt(raw, "limit.output")),
    asPos(getAt(raw, "top_provider.max_completion_tokens")),
  );

  const thinkingFormat = pick(asStr(caps?.thinkingFormat), asStr(raw.thinkingFormat), asStr(raw.thinking_format));
  const thinkingCanDisable = pick(
    asBool(caps?.thinkingCanDisable),
    asBool(raw.thinkingCanDisable),
    asBool(raw.thinking_can_disable),
  );
  const thinkingEffortSupported = pick(
    asBool(caps?.thinkingEffortSupported),
    asBool(raw.thinkingEffortSupported),
    asBool(raw.thinking_effort_supported),
  );

  return { reasoning, vision, contextWindow, maxTokens, thinkingFormat, thinkingCanDisable, thinkingEffortSupported };
}

/** Per-token price fields -> per-million-token cost. */
function extractUpstreamCost(raw: AnyRec): Cost | undefined {
  const perToken = (paths: string[]): number | undefined => pick(...paths.map((path) => asPos(getAt(raw, path))));
  const input = perToken(["pricing.prompt", "input_cost_per_token", "prompt_cost_per_token"]);
  const output = perToken(["pricing.completion", "output_cost_per_token", "completion_cost_per_token"]);
  if (input === undefined && output === undefined) return undefined;
  const million = (value: number | undefined) => (value ?? 0) * 1_000_000;
  return {
    input: million(input),
    output: million(output),
    cacheRead: million(perToken(["pricing.input_cache_read", "cache_read_input_token_cost"])),
    cacheWrite: million(perToken(["pricing.input_cache_write", "cache_creation_input_token_cost"])),
  };
}

// ---------------------------------------------------------------------------
// models.dev catalog
// ---------------------------------------------------------------------------

interface ReasoningOption { type?: string; values?: string[]; min?: number; max?: number }

interface ModelsDevModel {
  id?: string;
  reasoning?: boolean;
  reasoning_options?: ReasoningOption[];
  modalities?: { input?: string[]; output?: string[] };
  limit?: { context?: number; output?: number };
  canonical_model_id?: string;
}

let modelsDevCache: { at: number; data: Record<string, ModelsDevModel> } | undefined;

async function getModelsDevCatalog(signal?: AbortSignal): Promise<Record<string, ModelsDevModel>> {
  if (modelsDevCache && Date.now() - modelsDevCache.at < MODELS_DEV_TTL_MS) return modelsDevCache.data;
  try {
    const timeout = AbortSignal.timeout(60_000);
    const response = await fetch(MODELS_DEV_URL, {
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const raw = (await response.json()) as Record<string, { models?: Record<string, ModelsDevModel> }>;
    const data: Record<string, ModelsDevModel> = {};
    const bareCandidates = new Map<string, { providerId: string; model: ModelsDevModel }[]>();
    const bareLower = new Map<string, { providerId: string; model: ModelsDevModel }[]>();
    const push = (
      map: Map<string, { providerId: string; model: ModelsDevModel }[]>,
      key: string,
      value: { providerId: string; model: ModelsDevModel },
    ) => {
      const bucket = map.get(key) ?? map.set(key, []).get(key)!;
      bucket.push(value);
    };
    for (const [providerId, provider] of Object.entries(raw)) {
      if (!isRec(provider) || !isRec(provider.models)) continue;
      for (const [modelId, value] of Object.entries(provider.models)) {
        if (!isRec(value)) continue;
        const entry = value as ModelsDevModel;
        const id = asStr(entry.id) ?? modelId;
        data[`${providerId}/${id}`] = entry;
        const canonical = asStr(entry.canonical_model_id);
        // Prefer the first-party entry (the provider named in canonical_model_id).
        if (canonical && (canonical.split("/")[0] === providerId || !data[canonical])) data[canonical] = entry;
        push(bareCandidates, id, { providerId, model: entry });
        const lower = id.toLowerCase();
        if (lower !== id) push(bareLower, lower, { providerId, model: entry });
      }
    }
    for (const [id, candidates] of bareCandidates) {
      const representative = pickDevRepresentative(candidates);
      if (representative) data[id] = representative;
    }
    for (const [id, candidates] of bareLower) {
      const representative = pickDevRepresentative(candidates);
      if (representative) data[id] = representative;
    }
    modelsDevCache = { at: Date.now(), data };
    return data;
  } catch (error) {
    console.warn(`model-sync: models.dev fetch failed: ${String(error)}`);
    return modelsDevCache?.data ?? {};
  }
}

/**
 * Pick one models.dev entry for a model id that appears under many providers.
 * Prefer the first-party entry (provider named in canonical_model_id); otherwise
 * fall back to the most common capability shape.
 */
export function pickDevRepresentative(candidates: { providerId: string; model: ModelsDevModel }[]): ModelsDevModel | undefined {
  if (candidates.length === 0) return undefined;
  const preferred = [
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "moonshot",
    "zhipu",
    "xai",
    "mistral",
    "meta",
  ];
  for (const p of preferred) {
    const match = candidates.find(
      (c) => c.providerId === p || c.model?.canonical_model_id?.startsWith(`${p}/`),
    );
    if (match) return match.model;
  }
  const firstParty = candidates.find(({ providerId, model }) => {
    const canonical = asStr(model.canonical_model_id);
    return canonical !== undefined && canonical.split("/")[0] === providerId;
  });
  if (firstParty) return firstParty.model;
  if (candidates.length === 1) return candidates[0].model;
  const withCanonical = candidates.find((c) => Boolean(c.model?.canonical_model_id));
  if (withCanonical) return withCanonical.model;
  const sorted = [...candidates].sort((a, b) => {
    const ctxA = asPos(a.model.limit?.context) ?? 0;
    const ctxB = asPos(b.model.limit?.context) ?? 0;
    return ctxB - ctxA;
  });
  return sorted[0].model;
}

const BRACKET_TAG_RE = /\[[^\]]*\]/g;
const TAG_SUFFIX_RE = /(?::(?:free|thinking|preview|default)|@[^/]+)$/i;
const ROUTING_SUFFIX_RE = /-(?:openai-compact|compact|review|fast|wm|free)$/i;
const VARIANT_SUFFIX_RE = /-(?:high|low|medium|tiered|extra-low|extra|thinking|agent)$/i;
const ECO_PREFIX_RE = /^eco-/i;

export function generateLookupCandidates(id: string): string[] {
  const forms = new Set<string>([id]);

  if (id.includes("[")) {
    forms.add(id.replace(BRACKET_TAG_RE, ""));
  }

  if (id.includes("/")) {
    const parts = id.split("/");
    forms.add(parts[parts.length - 1]);
    if (parts.length > 2) {
      forms.add(parts.slice(1).join("/"));
    }
  }

  const result: string[] = [];
  const add = (c: string | undefined) => {
    if (!c) return;
    const clean = c.trim();
    if (clean && !result.includes(clean)) result.push(clean);
  };

  for (const form of forms) {
    add(form);
    const noBracket = form.replace(BRACKET_TAG_RE, "");
    add(noBracket);

    const noTag = noBracket.replace(TAG_SUFFIX_RE, "");
    add(noTag);

    const noRouting = noTag.replace(ROUTING_SUFFIX_RE, "");
    add(noRouting);

    const noVariant = noRouting.replace(VARIANT_SUFFIX_RE, "");
    add(noVariant);

    const noEco = noVariant.replace(ECO_PREFIX_RE, "");
    add(noEco);
  }

  for (const c of [...result]) {
    const hyphenated = c.replace(/(\d+)\.(\d+)/g, "$1-$2");
    add(hyphenated);
    const dotted = c.replace(/(\d+)-(\d+)/g, "$1.$2");
    add(dotted);
  }

  for (const c of [...result]) {
    if (c.endsWith("-preview")) add(c.slice(0, -"-preview".length));
    else add(`${c}-preview`);
    if (c.endsWith("-latest")) add(c.slice(0, -"-latest".length));
  }

  return result;
}

export function lookupModelsDev(catalog: Record<string, ModelsDevModel>, id: string): ModelsDevModel | undefined {
  const candidates = generateLookupCandidates(id);
  for (const candidate of candidates) {
    const direct = catalog[candidate] ?? catalog[candidate.toLowerCase()];
    if (direct) {
      const canonical = asStr(direct.canonical_model_id);
      return (canonical && catalog[canonical]) || direct;
    }
  }
  // Secondary fallback: if candidate has no slash, match any `${provider}/${candidate}`
  for (const candidate of candidates) {
    if (!candidate.includes("/")) {
      const suffix = `/${candidate.toLowerCase()}`;
      const matches: Array<{ providerId: string; model: ModelsDevModel }> = [];
      for (const [key, entry] of Object.entries(catalog)) {
        if (key.toLowerCase().endsWith(suffix)) {
          const providerId = key.split("/")[0].toLowerCase();
          const canonical = asStr(entry.canonical_model_id);
          matches.push({
            providerId,
            model: (canonical && catalog[canonical]) || entry,
          });
        }
      }
      if (matches.length > 0) {
        const representative = pickDevRepresentative(matches);
        if (representative) return representative;
      }
    }
  }
  return undefined;
}

/**
 * Translate models.dev `reasoning_options` into pi's `thinkingLevelMap`.
 *
 * - `{type:"effort", values:[...]}`: keep only the pi levels the model exposes.
 * - `{type:"toggle"}`: thinking is on/off, so hide the graded effort levels.
 * Any other shape (for example budget-only) leaves pi's defaults untouched.
 */
export function thinkingLevelMapFromDev(dev: ModelsDevModel | undefined): Record<string, string | null> | undefined {
  const options = dev?.reasoning_options;
  if (!Array.isArray(options) || options.length === 0) return undefined;
  const gradedLevels = ["minimal", "low", "medium", "high", "xhigh", "max"];
  const effort = options.find((option) => isRec(option) && option.type === "effort" && Array.isArray(option.values));
  const values = (effort?.values ?? []).filter((value): value is string => typeof value === "string");
  if (values.length > 0) {
    const map: Record<string, string | null> = {};
    for (const level of gradedLevels) map[level] = values.includes(level) ? level : null;
    return map;
  }
  if (options.some((option) => isRec(option) && option.type === "toggle")) {
    return { minimal: null, low: null, medium: null };
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// model construction
// ---------------------------------------------------------------------------

/** Last-resort heuristic for gpt-5.x / gpt-6.x models served behind gateways that report nothing. */
function gptFallback(id: string): { contextWindow: number; maxTokens: number; reasoning: boolean; vision: boolean } | undefined {
  const root = id.includes("/") ? id.slice(id.lastIndexOf("/") + 1) : id;
  const match = /^(gpt-[56](?:\.\d+)?)/.exec(root);
  if (!match) return undefined;
  const caps: Record<string, { contextWindow: number; maxTokens: number; reasoning: boolean; vision: boolean }> = {
    "gpt-5": { contextWindow: 400000, maxTokens: 128000, reasoning: false, vision: true },
    "gpt-5.1": { contextWindow: 400000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.2": { contextWindow: 400000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.3": { contextWindow: 400000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.4": { contextWindow: 1050000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.5": { contextWindow: 1050000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.6": { contextWindow: 1050000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-6": { contextWindow: 1050000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-6.1": { contextWindow: 1050000, maxTokens: 128000, reasoning: true, vision: true },
  };
  const key = Object.keys(caps).find((version) => root === version || root.startsWith(`${version}-`));
  if (!key) return undefined;
  if (root.includes("-chat-latest")) return { contextWindow: 128000, maxTokens: 16384, reasoning: true, vision: true };
  return caps[key];
}

/** Last-resort inference for gateways that expose no capability metadata. */
function inferReasoningFromId(id: string): boolean | undefined {
  return /(?:^|[-_/])(?:thinking|reasoning|reasoner|think|high|medium|low|tiered|extra)(?:[-_/]|\d|$)/i.test(id) ? true : undefined;
}

function defaultGptThinkingLevelMap(id: string): Record<string, string | null> | undefined {
  if (!/gpt-[56]/i.test(id)) return undefined;
  return {
    off: "none",
    minimal: null,
    low: "low",
    medium: "medium",
    high: "high",
    xhigh: "xhigh",
    max: /gpt-(?:5\.6|6)/i.test(id) ? "max" : null,
  };
}

export function buildModel(
  id: string,
  name: string | undefined,
  raw: AnyRec,
  dev: ModelsDevModel | undefined,
  override: ModelOverride | undefined,
): SyncedModel | undefined {
  const upstream = extractUpstream(raw);
  const heuristic = gptFallback(id);

  // Reasoning capability resolution:
  // 1. Explicit override wins unconditionally.
  // 2. Any authoritative positive signal confirms reasoning:
  //    upstream.reasoning === true, dev.reasoning === true,
  //    heuristic.reasoning === true, or inferReasoningFromId(id) === true.
  //    (Protects models like gpt-5.6-luna where gateways incorrectly report false).
  // 3. Fall back to upstream false / dev false / default false.
  let reasoning: boolean;
  if (override?.reasoning !== undefined) {
    reasoning = override.reasoning;
  } else if (
    upstream.reasoning === true ||
    dev?.reasoning === true ||
    heuristic?.reasoning === true ||
    inferReasoningFromId(id) === true
  ) {
    reasoning = true;
  } else if (upstream.reasoning === false) {
    reasoning = false;
  } else if (dev?.reasoning === false) {
    reasoning = false;
  } else {
    reasoning = false;
  }

  // Vision: an upstream `true` is trusted, but an upstream `false` is not —
  // some gateways mislabel vision-capable routes as text-only (verified live:
  // agnes/doubao routes answered image questions correctly while flagged
  // vision=false). When models.dev says the model accepts images, believe it.
  const devVision = includesImage(dev?.modalities?.input);
  const vision = pick(
    override?.vision,
    upstream.vision === true ? true : undefined,
    devVision === true ? true : undefined,
    upstream.vision,
    devVision,
    heuristic?.vision,
    false,
  ) as boolean;

  // ContextWindow and MaxTokens:
  // When an explicit override is provided, respect it unconditionally.
  // Otherwise take the maximum between upstream, models.dev, and heuristics,
  // preventing gateways from squashing official limits (e.g. 128k/8k placeholders
  // or 500k relay caps on 1M/2M models).
  const devContext = asPos(dev?.limit?.context);
  const contextWindow = override?.contextWindow ?? (
    upstream.contextWindow !== undefined || devContext !== undefined || heuristic?.contextWindow !== undefined
      ? Math.max(upstream.contextWindow ?? 0, devContext ?? 0, heuristic?.contextWindow ?? 0)
      : 128000
  );

  const devMaxTokens = asPos(dev?.limit?.output);
  const maxTokens = override?.maxTokens ?? (
    upstream.maxTokens !== undefined || devMaxTokens !== undefined || heuristic?.maxTokens !== undefined
      ? Math.max(upstream.maxTokens ?? 0, devMaxTokens ?? 0, heuristic?.maxTokens ?? 0)
      : 16384
  );

  if (contextWindow <= 0 || maxTokens <= 0) return undefined;

  // Thinking level map: restrict the levels pi offers to what the model exposes.
  // models.dev `reasoning_options` is the primary source; an override file wins
  // on top, and the upstream payload is only consulted when models.dev is silent.
  const devThinkingMap = thinkingLevelMapFromDev(dev);
  const thinkingLevelMap: Record<string, string | null> = { ...(devThinkingMap ?? {}) };
  if (override?.thinkingLevelMap) Object.assign(thinkingLevelMap, override.thinkingLevelMap);
  if (upstream.thinkingCanDisable === false) thinkingLevelMap.off = null;
  if (!devThinkingMap && upstream.thinkingEffortSupported === false) {
    thinkingLevelMap.minimal = null;
    thinkingLevelMap.low = null;
    thinkingLevelMap.medium = null;
  }
  if (reasoning && Object.keys(thinkingLevelMap).length === 0) {
    const gptThinkingMap = defaultGptThinkingLevelMap(id);
    if (gptThinkingMap) Object.assign(thinkingLevelMap, gptThinkingMap);
  }

  // Strong UI consistency invariant:
  // If active reasoning tiers exist (e.g. low/high/max), the model MUST be reasoning-capable.
  const hasActiveThinking = Object.entries(thinkingLevelMap).some(([k, v]) => k !== "off" && v !== null);
  if (override?.reasoning === undefined && hasActiveThinking) {
    reasoning = true;
  }

  // If reasoning is false, strip thinkingLevelMap completely to prevent UI mismatch
  // (e.g. "Thinking Level: 6" shown while reasoning checkbox is unchecked).
  if (!reasoning) {
    for (const key of Object.keys(thinkingLevelMap)) {
      delete thinkingLevelMap[key];
    }
  }

  // compat.thinkingFormat: only when the upstream value is one pi understands.
  const compat: AnyRec = {};
  if (upstream.thinkingFormat && PI_THINKING_FORMATS.has(upstream.thinkingFormat) && upstream.thinkingFormat !== "openai") {
    compat.thinkingFormat = upstream.thinkingFormat;
  }

  const model: SyncedModel = {
    id,
    name: name ?? id,
    reasoning,
    input: vision ? ["text", "image"] : ["text"],
    contextWindow,
    maxTokens,
    cost: extractUpstreamCost(raw) ?? ZERO_COST,
  };
  if (Object.keys(thinkingLevelMap).length > 0) model.thinkingLevelMap = thinkingLevelMap;
  if (Object.keys(compat).length > 0) model.compat = compat;
  return model;
}

/** Non-destructively merge newly synced model capabilities with existing model properties. */
export function mergeModelEntry(
  existing: AnyRec | undefined,
  generated: SyncedModel,
): SyncedModel {
  if (!existing || typeof existing !== "object") return generated;

  // 1. Preserve and union input modalities
  const inputSet = new Set<string>(generated.input || []);
  if (Array.isArray(existing.input)) {
    for (const item of existing.input) {
      if (typeof item === "string") inputSet.add(item);
    }
  }

  // 2. Preserve cost if generated is zero cost but existing had custom pricing
  const isZeroCost = (c: any) =>
    !c || (c.input === 0 && c.output === 0 && c.cacheRead === 0 && c.cacheWrite === 0);
  const cost =
    !isZeroCost(generated.cost)
      ? generated.cost
      : !isZeroCost(existing.cost)
      ? existing.cost
      : generated.cost;

  // 3. Merge compat options
  const compat = {
    ...(isRec(existing.compat) ? existing.compat : {}),
    ...(isRec(generated.compat) ? generated.compat : {}),
  };

  // 4. Merge thinkingLevelMap
  const thinkingLevelMap = generated.thinkingLevelMap
    ? {
        ...(isRec(existing.thinkingLevelMap) ? existing.thinkingLevelMap : {}),
        ...generated.thinkingLevelMap,
      }
    : undefined;

  // 5. Shallow copy any extra custom user fields from existing
  const merged: AnyRec = {
    ...existing,
    ...generated,
    input: Array.from(inputSet),
    cost,
  };

  // Preserve user custom name if previously customized
  if (
    typeof existing.name === "string" &&
    existing.name.trim() &&
    existing.name !== existing.id &&
    generated.name === generated.id
  ) {
    merged.name = existing.name;
  }

  if (thinkingLevelMap && Object.keys(thinkingLevelMap).length > 0) {
    merged.thinkingLevelMap = thinkingLevelMap;
  } else {
    delete merged.thinkingLevelMap;
  }

  if (Object.keys(compat).length > 0) {
    merged.compat = compat;
  } else {
    delete merged.compat;
  }

  return merged as SyncedModel;
}

export function isChatModel(raw: AnyRec, id: string): boolean {
  if (NON_CHAT_ID.test(id)) return false;
  const mode = asStr(raw.mode);
  if (mode && !/^(chat|responses|completion|model)$/i.test(mode)) return false;
  const endpoints = raw.supported_endpoints ?? raw.supported_endpoint_types;
  if (Array.isArray(endpoints) && endpoints.length > 0) {
    const chatty = endpoints.some((value) => typeof value === "string" && /chat|completions|responses|messages/i.test(value));
    if (!chatty) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// custom provider eligibility & protection
// ---------------------------------------------------------------------------

export const BUILTIN_PROVIDERS = new Set<string>([
  "amazon-bedrock",
  "ant-ling",
  "anthropic",
  "azure-openai-responses",
  "baseten",
  "cerebras",
  "cloudflare-ai-gateway",
  "cloudflare-workers-ai",
  "deepseek",
  "fireworks",
  "github-copilot",
  "google",
  "google-vertex",
  "groq",
  "huggingface",
  "kimi-coding",
  "meta",
  "minimax",
  "minimax-cn",
  "mistral",
  "moonshotai",
  "moonshotai-cn",
  "nvidia",
  "openai",
  "openai-codex",
  "opencode",
  "opencode-go",
  "openrouter",
  "qwen-token-plan",
  "qwen-token-plan-cn",
  "qwen-token-plan-individual",
  "radius",
  "together",
  "vercel-ai-gateway",
  "xai",
  "xiaomi",
  "xiaomi-token-plan-ams",
  "xiaomi-token-plan-cn",
  "xiaomi-token-plan-sgp",
  "zai",
  "zai-coding-cn",
]);

export interface ProviderEligibility {
  eligible: boolean;
  reason?: string;
}

export function isCustomProvider(
  name: string,
  provider: AnyRec,
  auth?: AnyRec,
): ProviderEligibility {
  if (provider.sync === false) {
    return { eligible: false, reason: "explicitly disabled (sync: false)" };
  }
  if (provider.sync === true) {
    const baseUrl = asStr(provider.baseUrl);
    if (!baseUrl) {
      return { eligible: false, reason: "missing baseUrl" };
    }
    return { eligible: true };
  }

  const baseUrl = asStr(provider.baseUrl);
  if (!baseUrl) {
    return { eligible: false, reason: "missing baseUrl" };
  }

  if (Boolean(provider.oauth)) {
    return { eligible: false, reason: "oauth provider" };
  }
  const cred = auth && isRec(auth[name]) ? auth[name] : undefined;
  if (cred && (cred.type === "oauth" || cred.access_token || cred.refresh_token || cred.tokens)) {
    return { eligible: false, reason: "oauth credential in auth.json" };
  }

  if (BUILTIN_PROVIDERS.has(name.toLowerCase())) {
    return { eligible: false, reason: "built-in provider" };
  }

  return { eligible: true };
}

// ---------------------------------------------------------------------------
// sync
// ---------------------------------------------------------------------------

export interface SyncOptions {
  dryRun?: boolean;
  targetProviders?: string[];
  signal?: AbortSignal;
  agentDir?: string;
}

export interface ModelDiff {
  id: string;
  type: "added" | "updated" | "unchanged";
  changes?: string[];
}

export interface ProviderDiff {
  provider: string;
  added: number;
  updated: number;
  unchanged: number;
  details: ModelDiff[];
}

export interface SyncResult {
  counts: Record<string, number>;
  errors: string[];
  providers: string[];
  skipped: string[];
  dryRun?: boolean;
  diffs?: ProviderDiff[];
}

export async function syncModels(options: SyncOptions = {}): Promise<SyncResult> {
  const agentDir = options.agentDir ?? AGENT_DIR;
  const modelsJsonPath = join(agentDir, "models.json");
  const authPath = join(agentDir, "auth.json");

  const modelsJson = await readModelsJson(modelsJsonPath); // throws on unreadable config -> nothing is written
  const auth = (await readJson<AnyRec>(authPath)) ?? {};
  const catalog = await getModelsDevCatalog(options.signal);

  const providers = { ...(modelsJson.providers ?? {}) };
  const counts: Record<string, number> = {};
  const errors: string[] = [];
  const synced: string[] = [];
  const skipped: string[] = [];
  const diffs: ProviderDiff[] = [];

  const targetNames = options.targetProviders && options.targetProviders.length > 0
    ? options.targetProviders
    : Object.keys(providers);

  for (const name of targetNames) {
    const provider = providers[name];
    if (!isRec(provider)) continue;
    const { eligible, reason } = isCustomProvider(name, provider, auth);
    if (!eligible) {
      skipped.push(`${name}${reason ? ` (${reason})` : ""}`);
      continue;
    }
    const baseUrl = asStr(provider.baseUrl)!;
    const api = asStr(provider.api) ?? "openai-completions";
    try {
      const apiKey = await resolveApiKey(provider, name, auth);
      const overrides = (await readJson<Record<string, ModelOverride>>(join(agentDir, `${name}-overrides.json`))) ?? {};
      const rawList = await fetchModelList(baseUrl, api, apiKey, provider.headers, options.signal);

      const existingMap = new Map<string, AnyRec>();
      if (Array.isArray(provider.models)) {
        for (const m of provider.models) {
          if (m && typeof m.id === "string") existingMap.set(m.id, m);
        }
      }

      const seen = new Set<string>();
      const nextModels: SyncedModel[] = [];
      const modelDiffs: ModelDiff[] = [];
      let added = 0;
      let updated = 0;
      let unchanged = 0;

      for (const raw of rawList) {
        const entry = normalizeEntry(raw);
        if (!entry || seen.has(entry.id)) continue;
        seen.add(entry.id);
        const override = overrides[entry.id];
        if (override?.filter === true) continue;
        if (!isChatModel(raw, entry.id)) continue;
        const normalized = buildModel(entry.id, entry.name, raw, lookupModelsDev(catalog, entry.id), override);
        if (!normalized) continue;

        const existing = existingMap.get(entry.id);
        const merged = mergeModelEntry(existing, normalized);

        if (!existing) {
          added += 1;
          modelDiffs.push({ id: entry.id, type: "added" });
        } else {
          const changes: string[] = [];
          if (existing.contextWindow !== merged.contextWindow) {
            changes.push(`context: ${existing.contextWindow} -> ${merged.contextWindow}`);
          }
          if (existing.maxTokens !== merged.maxTokens) {
            changes.push(`maxTokens: ${existing.maxTokens} -> ${merged.maxTokens}`);
          }
          if (existing.reasoning !== merged.reasoning) {
            changes.push(`reasoning: ${existing.reasoning} -> ${merged.reasoning}`);
          }
          if (JSON.stringify(existing.input) !== JSON.stringify(merged.input)) {
            changes.push(`input: ${JSON.stringify(existing.input)} -> ${JSON.stringify(merged.input)}`);
          }
          if (changes.length > 0) {
            updated += 1;
            modelDiffs.push({ id: entry.id, type: "updated", changes });
          } else {
            unchanged += 1;
            modelDiffs.push({ id: entry.id, type: "unchanged" });
          }
        }
        nextModels.push(merged);
      }

      nextModels.sort((a, b) => a.id.localeCompare(b.id));
      providers[name] = { ...provider, models: nextModels };
      counts[name] = nextModels.length;
      synced.push(name);
      diffs.push({ provider: name, added, updated, unchanged, details: modelDiffs });
    } catch (error) {
      counts[name] = 0;
      errors.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  // Backup and write
  if (!options.dryRun && synced.length > 0) {
    try {
      const currentRaw = await readFile(modelsJsonPath, "utf8");
      await writeFileAtomic(`${modelsJsonPath}.bak`, currentRaw);
    } catch {
      // ignore missing models.json on backup
    }
    await writeFileAtomic(modelsJsonPath, `${JSON.stringify({ ...modelsJson, providers }, null, 2)}\n`);
  }

  return { counts, errors, providers: synced, skipped, dryRun: options.dryRun, diffs };
}

export default function (pi: ExtensionAPI) {
  const handler = async (args: string, ctx: ExtensionCommandContext) => {
    const isDryRun = /(?:^|\s)(?:--dry-run|-n|--preview)(?:\s|$)/i.test(args || "");
    const targetProviders = (args || "")
      .split(/\s+/)
      .map((s) => s.trim())
      .filter((s) => s && !s.startsWith("-"));

    ctx.ui?.notify?.(
      isDryRun ? "Simulating model catalog refresh (dry-run)…" : "Refreshing model catalogs…",
      "info",
    );
    try {
      const result = await syncModels({ dryRun: isDryRun, targetProviders });
      const summary = result.providers
        .map((name) => {
          const diff = result.diffs?.find((d) => d.provider === name);
          if (diff) {
            return `${name}: ${result.counts[name] ?? 0} models (+${diff.added}, ~${diff.updated}, =${diff.unchanged})`;
          }
          return `${name} ${result.counts[name] ?? 0}`;
        })
        .join("; ") || "(none)";
      const skippedSummary = result.skipped.length > 0 ? ` [Skipped: ${result.skipped.join(", ")}]` : "";

      if (result.errors.length > 0) {
        ctx.ui?.notify?.(
          `${isDryRun ? "[Dry-run] " : ""}Synced: ${summary}${skippedSummary}\nErrors:\n${result.errors.join("\n")}`,
          "warning",
        );
      } else {
        ctx.ui?.notify?.(
          isDryRun
            ? `[Dry-run Preview] ${summary}${skippedSummary}\n(No files were modified)`
            : `models.json updated: ${summary}${skippedSummary} (backup created)`,
          "info",
        );
      }
    } catch (error) {
      ctx.ui?.notify?.(`Sync failed: ${error instanceof Error ? error.message : String(error)}`, "error");
    }
  };

  pi.registerCommand("refresh-custom-models", {
    description: "Discover models for custom providers in models.json (supports [provider] and --dry-run)",
    handler,
  });

  pi.registerCommand("refresh-custom-provider-models", {
    description: "Alias for /refresh-custom-models",
    handler,
  });
}
