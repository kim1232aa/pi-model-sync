# pi-model-sync

A generic model-catalog sync extension for [pi](https://pi.dev).

Add any provider to pi (or the pi-web Models panel) with a `baseUrl`, then run
`/refresh-custom-models`. The extension fetches that provider's model list,
matches each model's capabilities, and rewrites the provider's `models` array in
`~/.pi/agent/models.json`.

Nothing is hardcoded per provider, and it has no dependency on any particular
gateway or machine setup.

## Install

```bash
pi install git:github.com/kim1232aa/pi-model-sync
```

Or copy `extensions/model-sync.ts` into `~/.pi/agent/extensions/`.

## Usage

```
/refresh-custom-models
```

Works in both the pi CLI and the pi-web chat. Reload the session (`/reload`)
after installing or changing the extension.

## How it works

### Which providers get synced

Every provider in `models.json` that declares a **`baseUrl`** is synced. A
provider without `baseUrl` (for example a `compat`-only override of a built-in
provider) is left untouched.

Only the provider's `models` array is replaced; `name`, `baseUrl`, `apiKey`,
`headers`, `compat` and any other provider-level settings are preserved.

If a provider's model list cannot be fetched, that provider keeps its existing
models and the command reports the error. Other providers still sync.

### API key resolution

1. the provider's own `apiKey` — literal, `$ENV_VAR` / `${ENV_VAR}`, or `!command`
2. pi's stored credentials (`~/.pi/agent/auth.json`, written by `/login`)

### Model list

The endpoint is derived from `baseUrl`:

| `api` | endpoint | auth header |
| --- | --- | --- |
| `openai-completions`, `openai-responses`, … | `{baseUrl}/models` | `Authorization: Bearer` |
| `anthropic-messages` | `{baseUrl}/v1/models?limit=1000` | `x-api-key` + `anthropic-version` |
| `google-generative-ai` | `{baseUrl}/v1beta/models` | `x-goog-api-key` |

Response shapes understood: `{data:[…]}`, `{models:[…]}`, `{results:[…]}`,
`{items:[…]}`, a map of models, or a bare array.

### Capability matching

For each model, capabilities are resolved with this priority:

1. `<provider>-overrides.json` in the agent directory (manual override)
2. the **upstream model list payload** — many field names are recognised
   (`capabilities.*`, `modalities.input`, `limit.context/output`,
   `context_length`, `max_input_tokens` / `max_output_tokens`,
   `input_cost_per_token`, OpenRouter's `architecture` / `pricing` /
   `top_provider`, …)
3. the **models.dev catalog** (`https://models.dev/api.json`), matched by
   `provider/model`, bare model id, and `canonical_model_id`
4. a small gpt-5.x heuristic, an id-based reasoning hint, then defaults

Because the upstream payload describes what *that endpoint actually serves*, it
wins over models.dev for reasoning / vision / context window / max output when
both are present.

### Thinking levels

`thinkingLevelMap` is derived from models.dev's `reasoning_options`:

- `{"type":"effort","values":[…]}` → only the levels the model exposes are kept
  (for example `["low","high","max"]` hides `minimal`, `medium` and `xhigh`).
- `{"type":"toggle"}` → thinking is on/off, so the graded effort levels are hidden.
- a budget-only model leaves pi's defaults alone.

When models.dev has no `reasoning_options` for a model, the upstream
`thinkingCanDisable` / `thinkingEffortSupported` fields are used instead.
`thinkingFormat` is mapped into `compat.thinkingFormat` when it is a value pi
understands.

### Filtering

Model ids that are clearly not chat models (embeddings, rerankers, speech,
image/video/3d generation, upscalers, …) are skipped, as are entries whose
`mode` or `supported_endpoints` are clearly non-chat. Add an explicit
`{"filter": true}` entry to `<provider>-overrides.json` to drop a model manually.

## Overrides file

Optional, `<provider>-overrides.json` next to `models.json`:

```json
{
  "some-model-id": {
    "reasoning": true,
    "vision": false,
    "contextWindow": 200000,
    "maxTokens": 65536,
    "thinkingLevelMap": { "minimal": null, "low": "low", "high": "high" },
    "filter": false
  }
}
```

## Notes

- `models.json` is strictly validated by pi. This extension only ever writes
  fields pi knows (`id`, `name`, `reasoning`, `thinkingLevelMap`, `input`,
  `cost`, `contextWindow`, `maxTokens`, `compat`).
- A successful sync replaces the provider's model list, so models that no longer
  exist upstream are removed.
- models.dev data is cached for 6 hours per process.

## License

MIT
