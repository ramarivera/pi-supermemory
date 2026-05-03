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
- `PI_SUPERMEMORY_CONTAINER_TAG` or `SUPERMEMORY_CONTAINER_TAG`
- `SUPERMEMORY_API_BASE_URL`
- `PI_SUPERMEMORY_MAX_RECALL`
- `PI_SUPERMEMORY_AUTO_RECALL`
- `PI_SUPERMEMORY_AUTO_CAPTURE`

## Behavior

- Injects relevant Supermemory search results into Pi context before the model runs.
- Captures completed user/assistant turns back to the same Supermemory container.
- Registers tools:
  - `supermemory_search`
  - `supermemory_save`
  - `supermemory_status`
- Registers command:
  - `/supermemory status`
  - `/supermemory search <query>`
  - `/supermemory save <content>`

## Local development

```sh
npm install
npm test
npm run typecheck
```

The repo includes `.pi/extensions/pi-supermemory/index.ts` so Pi's `DefaultResourceLoader` can discover the extension locally.
