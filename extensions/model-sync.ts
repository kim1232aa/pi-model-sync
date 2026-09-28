import { execFile } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

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
const NON_CHAT_ID = /(?:^openai-compatible-chat|(?:^|\/)no-think\/|embedding|rerank|transcri|\btts\b|\baudio\b|realtime|background-task|image|video|seedream|seedance|seededit|seed3d|hitem3d|hyper3d|happyhorse|(?:^|[-_/])(?:i2v|t2v|flf2v|r2v)(?:[-_/]|$))/i;

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
// upstream endpoint
// ---------------------------------------------------------------------------

function normalizeBaseUrl(baseUrl: string, api: string): string {
  let url = baseUrl.replace(/\/+$/, "");
  if (api === "anthropic-messages" && !/\/v\d+(?:beta)?$/i.test(url)) url += "/v1";
  if (api === "google-generative-ai" && !/\/v\d+(?:beta)?$/i.test(url)) url += "/v1beta";
  return url;
}

function modelsEndpoint(baseUrl: string, api: string): string {
  const base = normalizeBaseUrl(baseUrl, api);
  const url = /\/models$/i.test(base) ? base : `${base}/models`;
  // Anthropic defaults to a small page size; ask for a large one.
  return api === "anthropic-messages" ? `${url}?limit=1000` : url;
}

function requestHeaders(api: string, apiKey: string | undefined, extra: unknown): Record<string, string> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (isRec(extra)) {
    for (const [key, value] of Object.entries(extra)) if (typeof value === "string") headers[key] = value;
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

async function fetchModelList(baseUrl: string, api: string, apiKey: string | undefined, extraHeaders: unknown): Promise<AnyRec[]> {
  const url = modelsEndpoint(baseUrl, api);
  const response = await fetch(url, {
    headers: requestHeaders(api, apiKey, extraHeaders),
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
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

function extractUpstream(raw: AnyRec): UpstreamCapabilities {
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
    const response = await fetch(MODELS_DEV_URL, { signal, headers: { Accept: "application/json" } });
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
function pickDevRepresentative(candidates: { providerId: string; model: ModelsDevModel }[]): ModelsDevModel | undefined {
  if (candidates.length === 0) return undefined;
  const firstParty = candidates.find(({ providerId, model }) => {
    const canonical = asStr(model.canonical_model_id);
    return canonical !== undefined && canonical.split("/")[0] === providerId;
  });
  if (firstParty) return firstParty.model;
  if (candidates.length === 1) return candidates[0].model;
  const groups = new Map<string, { count: number; model: ModelsDevModel }>();
  for (const { model } of candidates) {
    const key = JSON.stringify({
      reasoning: model.reasoning,
      reasoning_options: model.reasoning_options,
      modalities: model.modalities,
      limit: model.limit,
    });
    const group = groups.get(key) ?? { count: 0, model };
    group.count += 1;
    groups.set(key, group);
  }
  const sorted = [...groups.values()].sort((a, b) => b.count - a.count);
  if (sorted[1] && sorted[1].count === sorted[0].count) return undefined;
  return sorted[0].model;
}

function lookupModelsDev(catalog: Record<string, ModelsDevModel>, id: string): ModelsDevModel | undefined {
  const bare = id.includes("/") ? id.slice(id.lastIndexOf("/") + 1) : id;
  const direct = catalog[id] ?? catalog[bare] ?? catalog[bare.toLowerCase()];
  if (!direct) return undefined;
  // Follow canonical_model_id to the first-party entry when available.
  const canonical = asStr(direct.canonical_model_id);
  return (canonical && catalog[canonical]) || direct;
}

/**
 * Translate models.dev `reasoning_options` into pi's `thinkingLevelMap`.
 *
 * - `{type:"effort", values:[...]}`: keep only the pi levels the model exposes.
 * - `{type:"toggle"}`: thinking is on/off, so hide the graded effort levels.
 * Any other shape (for example budget-only) leaves pi's defaults untouched.
 */
function thinkingLevelMapFromDev(dev: ModelsDevModel | undefined): Record<string, string | null> | undefined {
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

/** Last-resort heuristic for gpt-5.x models served behind gateways that report nothing. */
function gptFallback(id: string): { contextWindow: number; maxTokens: number; reasoning: boolean; vision: boolean } | undefined {
  const root = id.includes("/") ? id.slice(id.lastIndexOf("/") + 1) : id;
  const match = /^(gpt-5(?:\.\d+)?)/.exec(root);
  if (!match) return undefined;
  const caps: Record<string, { contextWindow: number; maxTokens: number; reasoning: boolean; vision: boolean }> = {
    "gpt-5": { contextWindow: 400000, maxTokens: 128000, reasoning: false, vision: true },
    "gpt-5.1": { contextWindow: 400000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.2": { contextWindow: 400000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.3": { contextWindow: 400000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.4": { contextWindow: 1050000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.5": { contextWindow: 1050000, maxTokens: 128000, reasoning: true, vision: true },
    "gpt-5.6": { contextWindow: 1050000, maxTokens: 128000, reasoning: true, vision: true },
  };
  const key = Object.keys(caps).find((version) => root === version || root.startsWith(`${version}-`));
  if (!key) return undefined;
  if (root.includes("-chat-latest")) return { contextWindow: 128000, maxTokens: 16384, reasoning: true, vision: true };
  return caps[key];
}

/** Last-resort inference for gateways that expose no capability metadata. */
function inferReasoningFromId(id: string): boolean | undefined {
  return /(?:^|[-_/])(?:thinking|reasoning|reasoner|think)(?:[-_/]|\d|$)/i.test(id) ? true : undefined;
}

function defaultGptThinkingLevelMap(id: string): Record<string, string | null> | undefined {
  if (!/gpt-5/i.test(id)) return undefined;
  return {
    off: "none",
    minimal: null,
    low: "low",
    medium: "medium",
    high: "high",
    xhigh: "xhigh",
    max: /gpt-5\.6/i.test(id) ? "max" : null,
  };
}

function buildModel(
  id: string,
  name: string | undefined,
  raw: AnyRec,
  dev: ModelsDevModel | undefined,
  override: ModelOverride | undefined,
): SyncedModel | undefined {
  const upstream = extractUpstream(raw);
  const heuristic = gptFallback(id);

  const reasoning = pick(
    override?.reasoning,
    upstream.reasoning,
    dev?.reasoning,
    heuristic?.reasoning,
    inferReasoningFromId(id),
    false,
  ) as boolean;

  const vision = pick(
    override?.vision,
    upstream.vision,
    includesImage(dev?.modalities?.input),
    heuristic?.vision,
    false,
  ) as boolean;

  const contextWindow = pick(
    override?.contextWindow,
    upstream.contextWindow,
    asPos(dev?.limit?.context),
    heuristic?.contextWindow,
    128000,
  ) as number;

  const maxTokens = pick(
    override?.maxTokens,
    upstream.maxTokens,
    asPos(dev?.limit?.output),
    heuristic?.maxTokens,
    16384,
  ) as number;

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
  if (upstream.reasoning === undefined && upstream.thinkingEffortSupported === undefined && !dev && heuristic) {
    Object.assign(thinkingLevelMap, defaultGptThinkingLevelMap(id) ?? {});
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

function isChatModel(raw: AnyRec, id: string): boolean {
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
// sync
// ---------------------------------------------------------------------------

interface SyncResult {
  counts: Record<string, number>;
  errors: string[];
  providers: string[];
  skipped: string[];
}

async function syncModels(signal?: AbortSignal): Promise<SyncResult> {
  const modelsJson = (await readJson<{ providers?: Record<string, AnyRec> }>(MODELS_JSON_PATH)) ?? {};
  const auth = (await readJson<AnyRec>(AUTH_PATH)) ?? {};
  const catalog = await getModelsDevCatalog(signal);

  const providers = { ...(modelsJson.providers ?? {}) };
  const counts: Record<string, number> = {};
  const errors: string[] = [];
  const synced: string[] = [];
  const skipped: string[] = [];

  for (const [name, provider] of Object.entries(providers)) {
    if (!isRec(provider)) continue;
    const baseUrl = asStr(provider.baseUrl);
    if (!baseUrl) {
      skipped.push(name);
      continue;
    }
    const api = asStr(provider.api) ?? "openai-completions";
    try {
      const apiKey = await resolveApiKey(provider, name, auth);
      const overrides = (await readJson<Record<string, ModelOverride>>(join(AGENT_DIR, `${name}-overrides.json`))) ?? {};
      const rawList = await fetchModelList(baseUrl, api, apiKey, provider.headers);

      const seen = new Set<string>();
      const models: SyncedModel[] = [];
      for (const raw of rawList) {
        const entry = normalizeEntry(raw);
        if (!entry || seen.has(entry.id)) continue;
        seen.add(entry.id);
        const override = overrides[entry.id];
        if (override?.filter === true) continue;
        if (!isChatModel(raw, entry.id)) continue;
        const model = buildModel(entry.id, entry.name, raw, lookupModelsDev(catalog, entry.id), override);
        if (model) models.push(model);
      }

      models.sort((a, b) => a.id.localeCompare(b.id));
      providers[name] = { ...provider, models };
      counts[name] = models.length;
      synced.push(name);
    } catch (error) {
      counts[name] = 0;
      errors.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  await writeFile(MODELS_JSON_PATH, `${JSON.stringify({ providers }, null, 2)}\n`, { mode: 0o600 });
  return { counts, errors, providers: synced, skipped };
}

export default function (pi: ExtensionAPI) {
  pi.registerCommand("refresh-custom-models", {
    description: "Discover models for every models.json provider with a baseUrl and rewrite their model lists",
    handler: async (_args, ctx) => {
      ctx.ui.notify("Refreshing model catalogs…", "info");
      try {
        const result = await syncModels();
        const summary = result.providers.map((name) => `${name} ${result.counts[name] ?? 0}`).join(", ") || "(none)";
        if (result.errors.length > 0) {
          ctx.ui.notify(`Synced: ${summary}\nErrors:\n${result.errors.join("\n")}`, "warning");
        } else {
          ctx.ui.notify(`models.json updated: ${summary}`, "info");
        }
      } catch (error) {
        ctx.ui.notify(`Sync failed: ${error instanceof Error ? error.message : String(error)}`, "error");
      }
    },
  });
}
