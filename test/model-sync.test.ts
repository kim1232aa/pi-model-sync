// Run: npm test  (node --experimental-strip-types --test test/model-sync.test.ts)
import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  buildModel,
  extractUpstream,
  isChatModel,
  readModelsJson,
  stripJsonComments,
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

