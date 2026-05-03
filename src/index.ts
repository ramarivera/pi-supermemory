import { Type, type UserMessage } from "@mariozechner/pi-ai";
import type {
  ContextEvent,
  ExtensionAPI,
  ExtensionCommandContext,
  InputEvent,
  ToolDefinition,
  TurnEndEvent,
} from "@mariozechner/pi-coding-agent";

const DEFAULT_API_BASE_URL = "https://api.supermemory.ai";
const DEFAULT_CONTAINER_TAG = "ramiro-dev-memory";
const DEFAULT_MAX_RECALL = 5;
const EXTENSION_SOURCE = "pi-supermemory";

const EMPTY_SCHEMA = Type.Object({}, { additionalProperties: false });
const SEARCH_SCHEMA = Type.Object(
  {
    query: Type.String({ description: "Query to search in Supermemory." }),
    limit: Type.Optional(Type.Number({ description: "Maximum memories to return." })),
  },
  { additionalProperties: false },
);
const SAVE_SCHEMA = Type.Object(
  {
    content: Type.String({ description: "Memory content to save." }),
    is_static: Type.Optional(Type.Boolean({ description: "Whether Supermemory should treat this as static memory." })),
  },
  { additionalProperties: false },
);

type SearchParams = {
  query: string;
  limit?: number;
};

type SaveParams = {
  content: string;
  is_static?: boolean;
};

type TextToolResult<TDetails = unknown> = {
  content: Array<{ type: "text"; text: string }>;
  details: TDetails;
};

export type PiSupermemoryOptions = {
  apiKey?: string;
  apiBaseUrl?: string;
  containerTag?: string;
  commandName?: string;
  toolNamePrefix?: string;
  maxRecall?: number;
  autoRecall?: boolean;
  autoCapture?: boolean;
  client?: SupermemoryClient;
  clock?: () => number;
};

export type SupermemorySearchResult = {
  id?: string;
  content: string;
  score?: number;
  metadata?: Record<string, unknown>;
};

export type SupermemorySaveResult = {
  ok: boolean;
  status: number;
  id?: string;
  response?: unknown;
};

export interface SupermemoryClient {
  search(query: string, options?: { limit?: number }): Promise<SupermemorySearchResult[]>;
  save(content: string, options?: { isStatic?: boolean; metadata?: Record<string, unknown> }): Promise<SupermemorySaveResult>;
}

type Config = {
  apiKey: string | undefined;
  apiBaseUrl: string;
  containerTag: string;
  commandName: string;
  toolNamePrefix: string;
  maxRecall: number;
  autoRecall: boolean;
  autoCapture: boolean;
  clock: () => number;
};

type TextishMessage = {
  role?: string;
  content?: unknown;
  timestamp?: number;
  [key: string]: unknown;
};

type SupermemoryApiMemory = {
  id?: string;
  content?: string;
  text?: string;
  score?: number;
  similarity?: number;
  metadata?: Record<string, unknown>;
  document?: {
    id?: string;
    content?: string;
    text?: string;
    metadata?: Record<string, unknown>;
  };
  memory?: {
    id?: string;
    content?: string;
    text?: string;
    metadata?: Record<string, unknown>;
  };
};

type SupermemorySearchResponse = {
  results?: SupermemoryApiMemory[];
  memories?: SupermemoryApiMemory[];
  documents?: SupermemoryApiMemory[];
  data?: SupermemoryApiMemory[] | { results?: SupermemoryApiMemory[]; memories?: SupermemoryApiMemory[] };
};

export class SupermemoryHttpClient implements SupermemoryClient {
  readonly #apiKey: string;
  readonly #apiBaseUrl: string;
  readonly #containerTag: string;
  readonly #fetch: typeof fetch;

  constructor(options: { apiKey: string; apiBaseUrl?: string; containerTag?: string; fetchImpl?: typeof fetch }) {
    this.#apiKey = options.apiKey;
    this.#apiBaseUrl = trimTrailingSlash(options.apiBaseUrl ?? DEFAULT_API_BASE_URL);
    this.#containerTag = options.containerTag ?? DEFAULT_CONTAINER_TAG;
    this.#fetch = options.fetchImpl ?? fetch;
  }

  async search(query: string, options: { limit?: number } = {}): Promise<SupermemorySearchResult[]> {
    const response = await this.#request<SupermemorySearchResponse>("/v4/search", {
      q: query,
      containerTag: this.#containerTag,
      limit: options.limit ?? DEFAULT_MAX_RECALL,
      searchMode: "hybrid",
    });
    return normalizeSearchResponse(response);
  }

  async save(
    content: string,
    options: { isStatic?: boolean; metadata?: Record<string, unknown> } = {},
  ): Promise<SupermemorySaveResult> {
    const response = await this.#fetch(`${this.#apiBaseUrl}/v4/memories`, {
      method: "POST",
      headers: this.#headers(),
      body: JSON.stringify({
        containerTag: this.#containerTag,
        memories: [
          {
            content,
            isStatic: options.isStatic ?? false,
            metadata: {
              sm_source: EXTENSION_SOURCE,
              ...options.metadata,
            },
          },
        ],
      }),
    });

    const body = await readJson(response);
    if (!response.ok) {
      throw new Error(`Supermemory save failed with HTTP ${response.status}: ${JSON.stringify(body)}`);
    }
    const id = extractId(body);
    return {
      ok: true,
      status: response.status,
      ...(id ? { id } : {}),
      response: body,
    };
  }

  async #request<TResponse>(path: string, body: Record<string, unknown>): Promise<TResponse> {
    const response = await this.#fetch(`${this.#apiBaseUrl}${path}`, {
      method: "POST",
      headers: this.#headers(),
      body: JSON.stringify(body),
    });
    const json = await readJson(response);
    if (!response.ok) {
      throw new Error(`Supermemory request failed with HTTP ${response.status}: ${JSON.stringify(json)}`);
    }
    return json as TResponse;
  }

  #headers(): Record<string, string> {
    return {
      Authorization: `Bearer ${this.#apiKey}`,
      "Content-Type": "application/json",
    };
  }
}

export function createSupermemoryExtension(options: PiSupermemoryOptions = {}) {
  const config = resolveConfig(options);
  const client =
    options.client ??
    (config.apiKey
      ? new SupermemoryHttpClient({
          apiKey: config.apiKey,
          apiBaseUrl: config.apiBaseUrl,
          containerTag: config.containerTag,
        })
      : undefined);

  let latestUserInput = "";
  let latestSavedFingerprint = "";

  return {
    register(pi: ExtensionAPI): void {
      const searchTool: ToolDefinition<typeof SEARCH_SCHEMA, { query: string; results: SupermemorySearchResult[] } | { error: string }, unknown> = {
        name: `${config.toolNamePrefix}supermemory_search`,
        label: "Supermemory Search",
        description: `Search the shared Supermemory container "${config.containerTag}".`,
        parameters: SEARCH_SCHEMA,
        async execute(_toolCallId, params) {
          if (!client) return makeTextResult({ error: "Supermemory is not configured. Set SUPERMEMORY_API_KEY." });
          const results = await client.search(params.query, { limit: clampLimit(params.limit, config.maxRecall) });
          return makeTextResult({ query: params.query, results });
        },
      };
      pi.registerTool(searchTool);

      const saveTool: ToolDefinition<typeof SAVE_SCHEMA, SupermemorySaveResult | { error: string }, unknown> = {
        name: `${config.toolNamePrefix}supermemory_save`,
        label: "Supermemory Save",
        description: `Save a durable memory into "${config.containerTag}".`,
        parameters: SAVE_SCHEMA,
        async execute(_toolCallId, params) {
          if (!client) return makeTextResult({ error: "Supermemory is not configured. Set SUPERMEMORY_API_KEY." });
          const result = await client.save(params.content, {
            isStatic: params.is_static ?? false,
            metadata: memoryMetadata(config, "manual_tool"),
          });
          return makeTextResult(result);
        },
      };
      pi.registerTool(saveTool);

      const statusTool: ToolDefinition<typeof EMPTY_SCHEMA, Record<string, unknown>, unknown> = {
        name: `${config.toolNamePrefix}supermemory_status`,
        label: "Supermemory Status",
        description: "Show Supermemory configuration for the Pi extension.",
        parameters: EMPTY_SCHEMA,
        async execute() {
          return makeTextResult(statusPayload(config, Boolean(client)));
        },
      };
      pi.registerTool(statusTool);

      pi.registerCommand(config.commandName, {
        description: "Search, save, or inspect Supermemory.",
        handler: async (argumentString: string, ctx: ExtensionCommandContext) => {
          const args = argumentString.trim().split(/\s+/).filter(Boolean);
          const [action, ...rest] = args;
          if (!action || action === "status") {
            await notify(ctx, JSON.stringify(statusPayload(config, Boolean(client)), null, 2));
            return;
          }
          if (!client) {
            await notify(ctx, "Supermemory is not configured. Set SUPERMEMORY_API_KEY.");
            return;
          }
          if (action === "search") {
            const query = rest.join(" ").trim();
            if (!query) {
              await notify(ctx, `Usage: /${config.commandName} search <query>`);
              return;
            }
            const results = await client.search(query, { limit: config.maxRecall });
            await notify(ctx, formatSearchResults(results));
            return;
          }
          if (action === "save") {
            const content = rest.join(" ").trim();
            if (!content) {
              await notify(ctx, `Usage: /${config.commandName} save <content>`);
              return;
            }
            await client.save(content, { metadata: memoryMetadata(config, "manual_command") });
            await notify(ctx, `Saved to Supermemory container "${config.containerTag}".`);
            return;
          }
          await notify(ctx, `Unknown /${config.commandName} action "${action}". Try status, search, or save.`);
        },
      });

      pi.on("input", (event: InputEvent) => {
        if (event.source === "extension") return;
        const input = extractText(event);
        if (input) latestUserInput = input;
      });

      pi.on("context", async (event: ContextEvent) => {
        if (!client || !config.autoRecall || !latestUserInput.trim()) return { messages: event.messages };
        const results = await client.search(latestUserInput, { limit: config.maxRecall });
        if (results.length === 0) return { messages: event.messages };
        const recallMessage: UserMessage = {
          role: "user",
          content: `Relevant Supermemory context from "${config.containerTag}":\n\n${formatSearchResults(results)}`,
          timestamp: config.clock(),
        };
        return { messages: [recallMessage, ...event.messages] };
      });

      pi.on("turn_end", async (event: TurnEndEvent) => {
        if (!client || !config.autoCapture) return;
        const assistantText = extractText(event.message);
        if (!latestUserInput.trim() || !assistantText.trim()) return;
        const content = `Pi coding-agent turn\n\nUser:\n${latestUserInput.trim()}\n\nAssistant:\n${assistantText.trim()}`;
        const fingerprint = `${latestUserInput.trim()}\n---\n${assistantText.trim()}`;
        if (fingerprint === latestSavedFingerprint) return;
        latestSavedFingerprint = fingerprint;
        await client.save(content, { metadata: memoryMetadata(config, "turn_end") });
      });
    },
  };
}

function resolveConfig(options: PiSupermemoryOptions): Config {
  return {
    apiKey:
      options.apiKey ??
      process.env.SUPERMEMORY_API_KEY ??
      process.env.SUPERMEMORY_CC_API_KEY ??
      process.env.SUPERMEMORY_OPENCLAW_API_KEY,
    apiBaseUrl: trimTrailingSlash(options.apiBaseUrl ?? process.env.SUPERMEMORY_API_BASE_URL ?? DEFAULT_API_BASE_URL),
    containerTag: options.containerTag ?? process.env.PI_SUPERMEMORY_CONTAINER_TAG ?? process.env.SUPERMEMORY_CONTAINER_TAG ?? DEFAULT_CONTAINER_TAG,
    commandName: options.commandName ?? "supermemory",
    toolNamePrefix: options.toolNamePrefix ?? "",
    maxRecall: options.maxRecall ?? parsePositiveInteger(process.env.PI_SUPERMEMORY_MAX_RECALL, DEFAULT_MAX_RECALL),
    autoRecall: options.autoRecall ?? parseBoolean(process.env.PI_SUPERMEMORY_AUTO_RECALL, true),
    autoCapture: options.autoCapture ?? parseBoolean(process.env.PI_SUPERMEMORY_AUTO_CAPTURE, true),
    clock: options.clock ?? Date.now,
  };
}

function normalizeSearchResponse(response: SupermemorySearchResponse): SupermemorySearchResult[] {
  const candidates = Array.isArray(response.results)
    ? response.results
    : Array.isArray(response.memories)
      ? response.memories
      : Array.isArray(response.documents)
        ? response.documents
        : Array.isArray(response.data)
          ? response.data
          : Array.isArray(response.data?.results)
            ? response.data.results
            : Array.isArray(response.data?.memories)
              ? response.data.memories
              : [];

  return candidates.flatMap((candidate) => {
    const nested = candidate.memory ?? candidate.document ?? candidate;
    const content = nested.content ?? nested.text ?? candidate.content ?? candidate.text;
    if (!content) return [];
    const id = nested.id ?? candidate.id;
    const score = candidate.score ?? candidate.similarity;
    const metadata = nested.metadata ?? candidate.metadata;
    return [
      {
        ...(id ? { id } : {}),
        content,
        ...(score === undefined ? {} : { score }),
        ...(metadata ? { metadata } : {}),
      },
    ];
  });
}

function formatSearchResults(results: SupermemorySearchResult[]): string {
  if (results.length === 0) return "No Supermemory results found.";
  return results
    .map((result, index) => {
      const score = typeof result.score === "number" ? ` score=${result.score.toFixed(3)}` : "";
      return `${index + 1}. ${result.content}${score}`;
    })
    .join("\n\n");
}

function extractText(value: unknown): string {
  if (typeof value === "string") return value;
  if (!isRecord(value)) return "";
  const message = value as TextishMessage;
  const content = message.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .flatMap((part) => {
        if (typeof part === "string") return [part];
        if (isRecord(part) && typeof part.text === "string") return [part.text];
        if (isRecord(part) && typeof part.thinking === "string") return [part.thinking];
        return [];
      })
      .join("\n")
      .trim();
  }
  if (typeof message.text === "string") return message.text;
  if (typeof message.input === "string") return message.input;
  if (Array.isArray(message.contentParts)) return extractText({ content: message.contentParts });
  return "";
}

function memoryMetadata(config: Config, captureMode: string): Record<string, unknown> {
  return {
    source: EXTENSION_SOURCE,
    capture_mode: captureMode,
    container_tag: config.containerTag,
    captured_at: new Date(config.clock()).toISOString(),
  };
}

function statusPayload(config: Config, configured: boolean): Record<string, unknown> {
  return {
    configured,
    containerTag: config.containerTag,
    apiBaseUrl: config.apiBaseUrl,
    autoRecall: config.autoRecall,
    autoCapture: config.autoCapture,
    maxRecall: config.maxRecall,
  };
}

function makeTextResult<TDetails>(details: TDetails): TextToolResult<TDetails> {
  return {
    content: [{ type: "text", text: typeof details === "string" ? details : JSON.stringify(details, null, 2) }],
    details,
  };
}

function clampLimit(value: number | undefined, fallback: number): number {
  if (!Number.isFinite(value) || value === undefined) return fallback;
  return Math.max(1, Math.min(25, Math.floor(value)));
}

function parsePositiveInteger(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  if (["1", "true", "yes", "on"].includes(value.toLowerCase())) return true;
  if (["0", "false", "no", "off"].includes(value.toLowerCase())) return false;
  return fallback;
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text.trim()) return {};
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { text };
  }
}

function extractId(value: unknown): string | undefined {
  if (!isRecord(value)) return undefined;
  if (typeof value.id === "string") return value.id;
  if (Array.isArray(value.ids) && typeof value.ids[0] === "string") return value.ids[0];
  if (Array.isArray(value.memories) && isRecord(value.memories[0]) && typeof value.memories[0].id === "string") return value.memories[0].id;
  if (isRecord(value.data) && typeof value.data.id === "string") return value.data.id;
  return undefined;
}

async function notify(ctx: ExtensionCommandContext, message: string): Promise<void> {
  const ui = ctx.ui as { notify?: (message: string) => Promise<void> | void } | undefined;
  if (ui?.notify) {
    await ui.notify(message);
    return;
  }
  console.log(message);
}

function piSupermemoryExtension(pi: ExtensionAPI): void {
  createSupermemoryExtension().register(pi);
}

export default piSupermemoryExtension;
