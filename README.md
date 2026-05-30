# pi-supermemory

Pi coding-agent extension that recalls and captures development memory through Supermemory.

The extension uses Supermemory's v4 API directly. Pi does not expose MCP servers natively, so this package complements the toolbox `mcporter` MCP bridge instead of replacing it.

## Configuration

Set an API key in the Pi runtime environment:

```sh
SUPERMEMORY_API_KEY=...
```

By default, the extension uses a generic Supermemory container:

```sh
PI_SUPERMEMORY_CONTAINER_TAG=pi-supermemory
```

Set `PI_SUPERMEMORY_CONTAINER_TAG` to your own shared memory container if you want Pi to write into the same namespace as your other agents.

Supported environment variables:

- `SUPERMEMORY_API_KEY`, `SUPERMEMORY_CC_API_KEY`, or `SUPERMEMORY_OPENCLAW_API_KEY`
- `PI_SUPERMEMORY_CONFIG`
- `PI_SUPERMEMORY_CONTAINER_TAG` or `SUPERMEMORY_CONTAINER_TAG`
- `PI_SUPERMEMORY_ENABLED`
- `SUPERMEMORY_API_BASE_URL`
- `PI_SUPERMEMORY_MAX_RECALL`
- `PI_SUPERMEMORY_AUTO_RECALL`
- `PI_SUPERMEMORY_AUTO_CAPTURE`
- `PI_SUPERMEMORY_CAPTURE_MODE` (`signal` by default, or `all` for legacy every-turn capture)

Env vars always win over file-based configuration.

## Policy files

The extension discovers and merges config files **from the current working directory all the way up to `~`**. At each directory level it checks (in order of specificity):

```text
.pi/supermemory.json
.pi/pi-supermemory.json
.pi/agent/pi-supermemory.json
```

Child directories override parent directories. Within the same directory, `agent/pi-supermemory.json` wins over `pi-supermemory.json`, which wins over `supermemory.json`.

Use `PI_SUPERMEMORY_CONFIG` to point at a single explicit file instead of hierarchical discovery.

### Example

```json
{
  "default": {
    "enabled": true,
    "containerTag": "pi-supermemory",
    "captureMode": "signal"
  },
  "directories": {
    "/workspace/app": {
      "containerTag": "app-memory"
    }
  },
  "models": {
    "local/no-memory-model": {
      "enabled": false
    },
    "openai-codex/gpt-5.5": {
      "containerTag": "codex-memory"
    }
  },
  "rules": [
    {
      "path": "/workspace/app",
      "modelPattern": "openai-codex/.*",
      "containerTag": "app-codex-memory",
      "permissions": "read-only"
    }
  ]
}
```

### Override precedence

```text
default -> longest matching directory -> matching model -> matching rule
```

Model overrides win over directory overrides. **Rules win over everything** (they combine path + model pattern). Env vars win over all file-based config.

### Rules

Rules are the most specific override. Each rule has:

- `path` — directory scope (applies to this directory and all subdirectories unless overridden)
- `modelPattern` — optional regex matched against `<provider>/<model>` identifiers
- `containerTag`, `enabled`, `maxRecall`, etc. — same fields as other overrides
- `permissions` — `"read-only"`, `"write-only"`, or `"read-write"` (default)

When `permissions` is set, it controls whether the extension can search (read) and/or save (write) memories:

- `read-only` — search works, save is blocked
- `write-only` — save works, search is blocked
- `read-write` — both work (default)

If a subfolder defines its own rule, it overrides the parent rule.

## Behavior

- Injects relevant Supermemory search results into Pi context before the model runs (if `autoRecall` is enabled and read is permitted).
- Captures completed user/assistant turns back to the same Supermemory container (if `autoCapture` is enabled and write is permitted).
- Uses `captureMode: "signal"` by default to reduce Supermemory UI clutter: explicit memory requests, durable preferences/decisions, and completed implementation summaries are captured, while low-signal command chatter is skipped.
- Skips assistant thinking blocks and strips injected Supermemory context before saving, so recalled memories do not recursively re-enter storage.
- Set `captureMode: "all"` to restore legacy every-turn capture for a project, model, or rule.
- Splits oversized direct-memory writes into ordered chunks under Supermemory's per-memory content limit.
- Reports auto-capture save failures as concise warnings instead of surfacing extension stack traces.
- Registers tools:
  - `supermemory_search`
  - `supermemory_save`
  - `supermemory_save_file`
  - `supermemory_status`
- Registers command:
  - `/supermemory status`
  - `/supermemory search <query>`
  - `/supermemory save <content>`
  - `/supermemory save-file <path> [containerTag]`

## Local development

```sh
npm install
npm test
npm run typecheck
```

The repo includes `.pi/extensions/pi-supermemory/index.ts` so Pi's `DefaultResourceLoader` can discover the extension locally.
