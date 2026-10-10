// Run: npm test  (node --experimental-strip-types --test test/model-sync.test.ts)
import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  BUILTIN_PROVIDERS,
  buildModel,
  extractUpstream,
  generateLookupCandidates,
  getCandidateModelUrls,
  isChatModel,
  isCustomProvider,
  lookupModelsDev,
  mergeModelEntry,
  pickDevRepresentative,
  readModelsJson,
  stableStringify,
  stripJsonComments,
  syncModels,
  thinkingLevelMapFromDev,
} from "../extensions/model-sync.ts";

test("models.json comments are stripped, URLs inside strings survive", () => {
  const text = '{\n  // note\n  "a": "http://x/v1", /* block */ "b": 1\n}';
  assert.deepEqual(JSON.parse(stripJsonComments(text)), { a: "http://x/v1", b: 1 });
});

test("unparseable models.json throws instead of being treated as empty", async () => {
  const dir = await mkdtemp(join(tmpdir(), "model-sync-"));
  await writeFile(join(dir, "models.json"), "{ broken");
  await assert.rejects(readModelsJson(join(dir, "models.json")));
  assert.deepEqual(await readModelsJson(join(dir, "missing.json")), {});
});

test("upstream capabilities: explicit false wins, camelCase + snake_case shapes", () => {
  const a = extractUpstream({ capabilities: { reasoning: false, vision: true, contextWindow: 400000, maxOutput: 128000 } });
  assert.deepEqual([a.reasoning, a.vision, a.contextWindow, a.maxTokens], [false, true, 400000, 128000]);
  const b = extractUpstream({ context_length: 200000, max_output_tokens: 8192, architecture: { input_modalities: ["text"] } });
  assert.deepEqual([b.vision, b.contextWindow, b.maxTokens], [false, 200000, 8192]);
});

test("models.dev reasoning_options -> thinkingLevelMap", () => {
  assert.deepEqual(thinkingLevelMapFromDev({ reasoning_options: [{ type: "effort", values: ["low", "high", "max"] }] }), {
    minimal: null, low: "low", medium: null, high: "high", xhigh: null, max: "max",
  });
  assert.deepEqual(thinkingLevelMapFromDev({ reasoning_options: [{ type: "toggle" }] }), { minimal: null, low: null, medium: null });
  assert.equal(thinkingLevelMapFromDev({ reasoning_options: [] }), undefined);
});

test("non-chat ids are filtered, vision chat models are not", () => {
  assert.equal(isChatModel({}, "text-embedding-3-large"), false);
  assert.equal(isChatModel({}, "grok-imagine-1"), false);
  assert.equal(isChatModel({}, "qwen3-vl-image-understanding"), true);
  assert.equal(isChatModel({}, "kimi-k2-video-chat"), true);
});

test("vision resolution: upstream true kept, upstream false + models.dev true widened", () => {
  // Case 1: upstream says true -> stays true even if dev is missing or false
  const m1 = buildModel("m1", undefined, { capabilities: { vision: true } }, undefined, undefined);
  assert.deepEqual(m1?.input, ["text", "image"]);

  // Case 2: upstream says false, but models.dev says image input -> widened to true
  const m2 = buildModel("m2", undefined, { capabilities: { vision: false } }, { modalities: { input: ["text", "image"] } }, undefined);
  assert.deepEqual(m2?.input, ["text", "image"]);

  // Case 3: upstream says false, models.dev says text only -> stays false
  const m3 = buildModel("m3", undefined, { capabilities: { vision: false } }, { modalities: { input: ["text"] } }, undefined);
  assert.deepEqual(m3?.input, ["text"]);

  // Case 4: override wins unconditionally
  const m4 = buildModel("m4", undefined, { capabilities: { vision: false } }, undefined, { vision: true });
  assert.deepEqual(m4?.input, ["text", "image"]);
});

test("variant candidates strip thinking tiers, tags, normalize versions and previews", () => {
  const cands = generateLookupCandidates("tokenharbor/claude-haiku-5.5:free");
  assert.ok(cands.includes("claude-haiku-5.5"));
  assert.ok(cands.includes("claude-haiku-5-5"));

  const agCands = generateLookupCandidates("claude-opus-5-5-high");
  assert.ok(agCands.includes("claude-opus-5-5"));
  assert.ok(agCands.includes("claude-opus-5-5-preview"));

  const geminiCands = generateLookupCandidates("gemini-3.1-pro");
  assert.ok(geminiCands.includes("gemini-3.1-pro-preview"));
});

test("lookupModelsDev matches variant IDs against base models.dev catalog", () => {
  const catalog: any = {
    "anthropic/claude-haiku-5-5": {
      id: "claude-haiku-5-5",
      canonical_model_id: "anthropic/claude-haiku-5-5",
      reasoning: true,
      limit: { context: 1000000, output: 128000 },
    },
    "google/gemini-3.6-flash": {
      id: "gemini-3.6-flash",
      canonical_model_id: "google/gemini-3.6-flash",
      reasoning: true,
      limit: { context: 1000000, output: 65536 },
    },
  };
  // Strip tier suffix -high
  const res1 = lookupModelsDev(catalog, "gemini-3.6-flash-high");
  assert.equal(res1?.id, "gemini-3.6-flash");

  // Strip tag :free and match normalized 5.5 -> 5-5
  const res2 = lookupModelsDev(catalog, "tokenharbor/claude-haiku-5.5:free");
  assert.equal(res2?.id, "claude-haiku-5-5");
});

test("contextWindow & maxTokens take Math.max to prevent placeholder squashing", () => {
  // Upstream reports dummy 128k/8k, dev reports official 1M/64k -> Math.max picks 1M/64k
  const m = buildModel(
    "gemini-3.6-flash-high",
    undefined,
    { inputTokenLimit: 128000, outputTokenLimit: 8192 },
    { limit: { context: 1000000, output: 65536 } } as any,
    undefined,
  );
  assert.equal(m?.contextWindow, 1000000);
  assert.equal(m?.maxTokens, 65536);
  assert.equal(m?.reasoning, true);
});

test("gpt-6 bare id heuristic fallback matches reasoning, vision, and 1M context", () => {
  const m = buildModel("gpt-6", undefined, {}, undefined, undefined);
  assert.equal(m?.contextWindow, 1050000);
  assert.equal(m?.maxTokens, 128000);
  assert.equal(m?.reasoning, true);
  assert.deepEqual(m?.input, ["text", "image"]);
  assert.equal(m?.thinkingLevelMap?.max, "max");
});

test("gemini multimodal chat models survive filter, image generators are filtered", () => {
  assert.equal(isChatModel({}, "gemini-3.1-flash-image"), true);
  assert.equal(isChatModel({}, "imagen-3.0-generate-002"), false);
  assert.equal(isChatModel({}, "gemini-image-preview"), false);
});

test("isCustomProvider: protects built-ins, oauth, missing baseUrl, allows custom relays", () => {
  // 1. Valid custom provider with baseUrl
  const custom = isCustomProvider("custom-relay", { baseUrl: "https://relay.example.com/v1" });
  assert.equal(custom.eligible, true);

  // 2. Missing baseUrl is skipped
  const noBase = isCustomProvider("my-provider", {});
  assert.equal(noBase.eligible, false);
  assert.match(noBase.reason ?? "", /missing baseUrl/);

  // 3. Built-in providers protected by default
  assert.ok(BUILTIN_PROVIDERS.has("anthropic"));
  assert.ok(BUILTIN_PROVIDERS.has("openai"));
  assert.ok(BUILTIN_PROVIDERS.has("github-copilot"));
  const anthropic = isCustomProvider("anthropic", { baseUrl: "https://my-proxy/v1" });
  assert.equal(anthropic.eligible, false);
  assert.match(anthropic.reason ?? "", /built-in provider/);

  // 4. Built-in provider can be explicitly allowed via sync: true
  const forceBuiltin = isCustomProvider("openai", { baseUrl: "https://my-proxy/v1", sync: true });
  assert.equal(forceBuiltin.eligible, true);

  // 5. Custom provider can be explicitly disabled via sync: false
  const disabledCustom = isCustomProvider("custom-relay", { baseUrl: "https://relay.example.com/v1", sync: false });
  assert.equal(disabledCustom.eligible, false);
  assert.match(disabledCustom.reason ?? "", /explicitly disabled/);

  // 6. OAuth provider via config is skipped
  const oauthConfig = isCustomProvider("github-copilot", { baseUrl: "https://api.github.com", oauth: true });
  assert.equal(oauthConfig.eligible, false);

  // 7. OAuth provider via auth.json credential is skipped
  const oauthAuth = isCustomProvider("custom-oauth", { baseUrl: "https://auth.example.com" }, { "custom-oauth": { type: "oauth" } });
  assert.equal(oauthAuth.eligible, false);
  assert.match(oauthAuth.reason ?? "", /oauth credential/);
});

test("reasoning UI consistency: upstream false overridden by heuristic/dev, thinkingLevelMap synchronized", () => {
  // Case A: gpt-5.6-luna where gateway incorrectly reports reasoning=false
  const lunaRaw = {
    id: "gpt-5.6-luna",
    capabilities: {
      vision: true,
      reasoning: false, // gateway misreport
      contextWindow: 1048576,
      maxOutput: 524288,
    },
  };
  const lunaModel = buildModel("gpt-5.6-luna", "gpt-5.6-luna", lunaRaw, undefined, undefined);
  assert.ok(lunaModel);
  // Must be reasoning=true because gpt-5.6 is a known reasoning model
  assert.equal(lunaModel.reasoning, true);
  // Must have matching thinkingLevelMap with max tier for gpt-5.6
  assert.ok(lunaModel.thinkingLevelMap);
  assert.equal(lunaModel.thinkingLevelMap.max, "max");

  // Case B: Non-reasoning model must NEVER retain thinkingLevelMap
  const gpt5Raw = {
    id: "gpt-5",
    capabilities: {
      reasoning: false,
      contextWindow: 400000,
      maxOutput: 128000,
    },
  };
  const gpt5Model = buildModel("gpt-5", "gpt-5", gpt5Raw, undefined, undefined);
  assert.ok(gpt5Model);
  assert.equal(gpt5Model.reasoning, false);
  assert.equal(gpt5Model.thinkingLevelMap, undefined);
});

test("contextWindow restoration: cx/gpt-6-astra 272k pricing threshold restored to 1M official limit", () => {
  const devEntry: ModelsDevModel = {
    id: "gpt-6-astra",
    canonical_model_id: "openai/gpt-6-astra",
    reasoning: true,
    limit: { context: 1050000, output: 128000 },
  };
  const rawGateway = {
    id: "cx/gpt-6-astra",
    capabilities: {
      contextWindow: 272000, // gateway squashed to long-context pricing cutoff
      maxOutput: 128000,
      reasoning: true,
      vision: true,
    },
  };
  const model = buildModel("cx/gpt-6-astra", "cx/gpt-6-astra", rawGateway, devEntry, undefined);
  assert.ok(model);
  // Must restore to 1050000 from devEntry / heuristic rather than 272000
  assert.equal(model.contextWindow, 1050000);
  assert.equal(model.maxTokens, 128000);
  assert.equal(model.reasoning, true);
});

test("getCandidateModelUrls: tests /v1/models probe and protocol fallbacks", () => {
  // 1. URL without version suffix tries /v1/models first, falls back to /models
  const withoutV1 = getCandidateModelUrls("https://relay.example.com");
  assert.deepEqual(withoutV1, [
    "https://relay.example.com/v1/models",
    "https://relay.example.com/models",
  ]);

  // 2. URL already containing /v1 targets /v1/models directly
  const withV1 = getCandidateModelUrls("https://relay.example.com/v1");
  assert.deepEqual(withV1, ["https://relay.example.com/v1/models"]);

  // 3. Anthropic protocol
  const anthropic = getCandidateModelUrls("https://api.anthropic.com", "anthropic-messages");
  assert.deepEqual(anthropic, ["https://api.anthropic.com/v1/models?limit=1000"]);

  // 4. Google Generative AI protocol
  const google = getCandidateModelUrls("http://relay.example.com", "google-generative-ai");
  assert.deepEqual(google, [
    "http://relay.example.com/v1beta/models",
    "http://relay.example.com/models",
  ]);
});

test("pickDevRepresentative: prefers first-party over aggregator entries", () => {
  const candidates = [
    { providerId: "openrouter", model: { canonical_model_id: "openai/o3-mini", limit: { context: 200000 } } as any },
    { providerId: "openai", model: { canonical_model_id: "openai/o3-mini", limit: { context: 200000 } } as any },
  ];
  const chosen = pickDevRepresentative(candidates);
  assert.equal(chosen?.canonical_model_id, "openai/o3-mini");
});

test("mergeModelEntry: preserves existing custom modalities, cost, and metadata non-destructively", () => {
  const existing = {
    id: "gemini-2.5-pro",
    name: "My Custom Gemini",
    input: ["text", "image", "audio"],
    cost: { input: 1.25, output: 5, cacheRead: 0.3, cacheWrite: 1.25 },
    compat: { customOption: true },
    userNote: "custom configuration",
  };
  const generated = {
    id: "gemini-2.5-pro",
    name: "gemini-2.5-pro",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 1048576,
    maxTokens: 65536,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    thinkingLevelMap: { low: "low", high: "high" },
  };

  const merged = mergeModelEntry(existing, generated);
  // Preserves union of input modalities including audio
  assert.ok(merged.input.includes("audio"));
  assert.ok(merged.input.includes("image"));
  assert.ok(merged.input.includes("text"));

  // Preserves custom user pricing over zero-cost default
  assert.equal(merged.cost.input, 1.25);

  // Preserves custom user name
  assert.equal(merged.name, "My Custom Gemini");

  // Preserves user custom compat and arbitrary properties
  assert.equal((merged.compat as any)?.customOption, true);
  assert.equal((merged as any)?.userNote, "custom configuration");

  // Takes updated context and thinking
  assert.equal(merged.contextWindow, 1048576);
  assert.equal(merged.thinkingLevelMap?.high, "high");
});

test("dry-run and backup safety: simulates diff without disk modification, creates .bak on write", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-sync-test-"));
  const modelsPath = join(dir, "models.json");
  const bakPath = join(dir, "models.json.bak");

  const initialConfig = {
    providers: {
      mockrelay: {
        baseUrl: "https://mock.example.com/v1",
        sync: false, // disable network fetch
      },
    },
  };
  await writeFile(modelsPath, JSON.stringify(initialConfig, null, 2));

  // 1. Dry run
  const dryResult = await syncModels({ dryRun: true, agentDir: dir });
  assert.equal(dryResult.dryRun, true);
  assert.deepEqual(dryResult.skipped, ["mockrelay (explicitly disabled (sync: false))"]);

  // Ensure no backup file was created during dry run
  const { stat } = await import("node:fs/promises");
  await assert.rejects(stat(bakPath));
});

test("stableStringify: ignores key order, distinguishes thinkingLevelMap changes", () => {
  assert.equal(stableStringify(undefined), undefined);
  assert.equal(
    stableStringify({ low: "low", high: "high", xhigh: null }),
    stableStringify({ xhigh: null, high: "high", low: "low" }),
  );
  assert.notEqual(
    stableStringify({ high: "high", xhigh: null }),
    stableStringify({ high: "high", xhigh: "xhigh" }),
  );
});
