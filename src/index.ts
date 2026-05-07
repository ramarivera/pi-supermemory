import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, normalize, resolve } from "node:path";
import { Type, type Model, type UserMessage } from "@mariozechner/pi-ai";
import type {
  ContextEvent,
  ExtensionAPI,
  ExtensionCommandContext,
  ExtensionContext,
  InputEvent,
  ToolDefinition,
  TurnEndEvent,
} from "@mariozechner/pi-coding-agent";

const DEFAULT_API_BASE_URL = "https://api.supermemory.ai";
const DEFAULT_CONTAINER_TAG = "pi-supermemory";
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
  configPath?: string;
  commandName?: string;
  toolNamePrefix?: string;
  enabled?: boolean;
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
  enabled: boolean;
  apiKey: string | undefined;
  apiBaseUrl: string;
  containerTag: string;
  commandName: string;
  toolNamePrefix: string;
  maxRecall: number;
  autoRecall: boolean;
  autoCapture: boolean;
  permissions: Permission | undefined;
  clock: () => number;
  configPath: string | undefined;
  matchedDirectory: string | undefined;
  matchedModel: string | undefined;
  matchedRule: string | undefined;
};

export type RuntimeConfig = Omit<Config, "clock">;

export type Permission = "read-only" | "write-only" | "read-write";

export type ConfigOverride = {
  enabled?: boolean;
  apiKey?: string;
  apiBaseUrl?: string;
  containerTag?: string;
  maxRecall?: number;
  autoRecall?: boolean;
  autoCapture?: boolean;
  permissions?: Permission;
};

export type DirectoryOverride = ConfigOverride & {
  path?: string;
};

export type Rule = {
  path: string;
  modelPattern?: string;
  containerTag?: string;
  enabled?: boolean;
  apiKey?: string;
  apiBaseUrl?: string;
  maxRecall?: number;
  autoRecall?: boolean;
  autoCapture?: boolean;
  permissions?: Permission;
};

export type SupermemoryConfigFile = {
  default?: ConfigOverride;
  directories?: Record<string, ConfigOverride> | DirectoryOverride[];
  models?: Record<string, ConfigOverride>;
  rules?: Rule[];
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
  document?: string | {
    id?: string;
    content?: string;
    text?: string;
    metadata?: Record<string, unknown>;
  };
  memory?: string | {
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
  const baseConfig = resolveBaseConfig(options);
  const explicitConfigPath = baseConfig.configPath;
  const explicitPolicy = explicitConfigPath ? loadConfigFile(explicitConfigPath) : undefined;

  function getPolicy(cwd: string | undefined): SupermemoryConfigFile | undefined {
    if (explicitConfigPath !== undefined) return explicitPolicy;
    if (!cwd) return undefined;
    return loadMergedPolicyForCwd(cwd);
  }

  let latestUserInput = "";
  let latestSavedFingerprint = "";

  return {
    register(pi: ExtensionAPI): void {
      const searchTool: ToolDefinition<typeof SEARCH_SCHEMA, { query: string; results: SupermemorySearchResult[] } | { error: string }, unknown> = {
        name: `${baseConfig.toolNamePrefix}supermemory_search`,
        label: "Supermemory Search",
        description: "Search the active Supermemory container.",
        parameters: SEARCH_SCHEMA,
        async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
          const config = resolveRuntimeConfig(baseConfig, getPolicy(ctx?.cwd), ctx);
          const client = clientForConfig(config, options.client);
          if (!config.enabled) return makeTextResult({ error: "Supermemory is disabled by configuration." });
          if (!canRead(config)) return makeTextResult({ error: "Supermemory search is disabled by configuration (write-only)." });
          if (!client) return makeTextResult({ error: "Supermemory is not configured. Set SUPERMEMORY_API_KEY." });
          const results = await client.search(params.query, { limit: clampLimit(params.limit, config.maxRecall) });
          return makeTextResult({ query: params.query, results });
        },
      };
      pi.registerTool(searchTool);

      const saveTool: ToolDefinition<typeof SAVE_SCHEMA, SupermemorySaveResult | { error: string }, unknown> = {
        name: `${baseConfig.toolNamePrefix}supermemory_save`,
        label: "Supermemory Save",
        description: "Save a durable memory into the active Supermemory container.",
        parameters: SAVE_SCHEMA,
        async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
          const config = resolveRuntimeConfig(baseConfig, getPolicy(ctx?.cwd), ctx);
          const client = clientForConfig(config, options.client);
          if (!config.enabled) return makeTextResult({ error: "Supermemory is disabled by configuration." });
          if (!canWrite(config)) return makeTextResult({ error: "Supermemory save is disabled by configuration (read-only)." });
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
        name: `${baseConfig.toolNamePrefix}supermemory_status`,
        label: "Supermemory Status",
        description: "Show Supermemory configuration for the Pi extension.",
        parameters: EMPTY_SCHEMA,
        async execute(_toolCallId, _params, _signal, _onUpdate, ctx) {
          const config = resolveRuntimeConfig(baseConfig, getPolicy(ctx?.cwd), ctx);
          return makeTextResult(statusPayload(config, Boolean(clientForConfig(config, options.client))));
        },
      };
      pi.registerTool(statusTool);

      pi.registerCommand(baseConfig.commandName, {
        description: "Search, save, or inspect Supermemory.",
        handler: async (argumentString: string, ctx: ExtensionCommandContext) => {
          const config = resolveRuntimeConfig(baseConfig, getPolicy(ctx.cwd), ctx);
          const client = clientForConfig(config, options.client);
          const args = argumentString.trim().split(/\s+/).filter(Boolean);
          const [action, ...rest] = args;
          if (!action || action === "status") {
            await notify(ctx, JSON.stringify(statusPayload(config, Boolean(client)), null, 2));
            return;
          }
          if (!config.enabled) {
            await notify(ctx, "Supermemory is disabled by configuration.");
            return;
          }
          if (!client) {
            await notify(ctx, "Supermemory is not configured. Set SUPERMEMORY_API_KEY.");
            return;
          }
          if (action === "search") {
            if (!canRead(config)) {
              await notify(ctx, "Supermemory search is disabled by configuration (write-only).");
              return;
            }
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
            if (!canWrite(config)) {
              await notify(ctx, "Supermemory save is disabled by configuration (read-only).");
              return;
            }
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

      pi.on("context", async (event: ContextEvent, ctx) => {
        const config = resolveRuntimeConfig(baseConfig, getPolicy(ctx?.cwd), ctx);
        const client = clientForConfig(config, options.client);
        if (!client || !config.autoRecall || !canRead(config) || !latestUserInput.trim()) return { messages: event.messages };
        const results = await client.search(latestUserInput, { limit: config.maxRecall });
        if (results.length === 0) return { messages: event.messages };
        const recallMessage: UserMessage = {
          role: "user",
          content: `Relevant Supermemory context from "${config.containerTag}":\n\n${formatSearchResults(results)}`,
          timestamp: config.clock(),
        };
        return { messages: [recallMessage, ...event.messages] };
      });

      pi.on("turn_end", async (event: TurnEndEvent, ctx) => {
        const config = resolveRuntimeConfig(baseConfig, getPolicy(ctx?.cwd), ctx);
        const client = clientForConfig(config, options.client);
        if (!client || !config.autoCapture || !canWrite(config)) return;
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

export function resolveSupermemoryConfig(input: {
  base?: Partial<RuntimeConfig>;
  policy?: SupermemoryConfigFile;
  cwd?: string;
  model?: Model<any> | { id?: string; name?: string; provider?: string; api?: string } | undefined;
}): RuntimeConfig {
  const base: RuntimeConfig = {
    enabled: input.base?.enabled ?? true,
    apiKey: input.base?.apiKey,
    apiBaseUrl: input.base?.apiBaseUrl ?? DEFAULT_API_BASE_URL,
    containerTag: input.base?.containerTag ?? DEFAULT_CONTAINER_TAG,
    commandName: input.base?.commandName ?? "supermemory",
    toolNamePrefix: input.base?.toolNamePrefix ?? "",
    maxRecall: input.base?.maxRecall ?? DEFAULT_MAX_RECALL,
    autoRecall: input.base?.autoRecall ?? true,
    autoCapture: input.base?.autoCapture ?? true,
    permissions: input.base?.permissions,
    configPath: input.base?.configPath,
    matchedDirectory: undefined,
    matchedModel: undefined,
    matchedRule: undefined,
  };
  const withDefault = mergeOverride(base, input.policy?.default);
  const directoryMatch = findDirectoryOverride(input.policy?.directories, input.cwd);
  const withDirectory = mergeOverride(withDefault, directoryMatch?.override);
  const modelMatch = findModelOverride(input.policy?.models, input.model);
  const withModel = mergeOverride(withDirectory, modelMatch?.override);
  const ruleMatch = findRuleOverride(input.policy?.rules, input.cwd, input.model);
  const withRule = mergeOverride(withModel, ruleMatch?.override);
  return {
    ...withRule,
    matchedDirectory: directoryMatch?.path,
    matchedModel: modelMatch?.key,
    matchedRule: ruleMatch ? `${ruleMatch.path}${ruleMatch.override.permissions ? ` [${ruleMatch.override.permissions}]` : ""}` : undefined,
  };
}

function resolveBaseConfig(options: PiSupermemoryOptions): Config {
  return {
    enabled: options.enabled ?? parseBoolean(process.env.PI_SUPERMEMORY_ENABLED, true),
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
    permissions: undefined,
    clock: options.clock ?? Date.now,
    configPath: options.configPath ?? process.env.PI_SUPERMEMORY_CONFIG ?? undefined,
    matchedDirectory: undefined,
    matchedModel: undefined,
    matchedRule: undefined,
  };
}

function resolveRuntimeConfig(baseConfig: Config, policy: SupermemoryConfigFile | undefined, ctx?: Partial<ExtensionContext>): Config {
  const resolved = resolveSupermemoryConfig({
    base: baseConfig,
    ...(policy === undefined ? {} : { policy }),
    ...(ctx?.cwd === undefined ? {} : { cwd: ctx.cwd }),
    ...(ctx?.model === undefined ? {} : { model: ctx.model }),
  });
  return { ...resolved, clock: baseConfig.clock };
}

function clientForConfig(config: Config | RuntimeConfig, injected: SupermemoryClient | undefined): SupermemoryClient | undefined {
  if (!config.enabled) return undefined;
  if (injected) return injected;
  if (!config.apiKey) return undefined;
  return new SupermemoryHttpClient({ apiKey: config.apiKey, apiBaseUrl: config.apiBaseUrl, containerTag: config.containerTag });
}

function loadConfigFile(configPath: string): SupermemoryConfigFile | undefined {
  if (!existsSync(configPath)) return undefined;
  const raw = readFileSync(configPath, "utf8");
  const parsed = JSON.parse(raw) as unknown;
  if (!isRecord(parsed)) {
    throw new Error(`Pi Supermemory config must be a JSON object: ${configPath}`);
  }
  return parsed as SupermemoryConfigFile;
}

function mergeOverride(config: RuntimeConfig, override: ConfigOverride | undefined): RuntimeConfig {
  if (!override) return config;
  const result: RuntimeConfig = { ...config };
  if (override.enabled !== undefined) result.enabled = override.enabled;
  if (override.apiKey !== undefined) result.apiKey = override.apiKey;
  if (override.apiBaseUrl !== undefined) result.apiBaseUrl = trimTrailingSlash(override.apiBaseUrl);
  if (override.containerTag !== undefined) result.containerTag = override.containerTag;
  if (override.maxRecall !== undefined) result.maxRecall = override.maxRecall;
  if (override.autoRecall !== undefined) result.autoRecall = override.autoRecall;
  if (override.autoCapture !== undefined) result.autoCapture = override.autoCapture;
  if (override.permissions !== undefined) result.permissions = override.permissions;
  return result;
}

function findDirectoryOverride(
  directories: SupermemoryConfigFile["directories"] | undefined,
  cwd: string | undefined,
): { path: string; override: ConfigOverride } | undefined {
  if (!directories || !cwd) return undefined;
  const cwdPath = normalize(resolve(cwd));
  const entries = Array.isArray(directories)
    ? directories.flatMap((entry) => {
        if (!entry.path) return [];
        const { path, ...override } = entry;
        return [{ path, override }];
      })
    : Object.entries(directories).map(([path, override]) => ({ path, override }));

  return entries
    .map((entry) => ({ path: normalize(resolve(entry.path)), override: entry.override }))
    .filter((entry) => isDirectoryMatch(cwdPath, entry.path))
    .sort((a, b) => b.path.length - a.path.length)[0];
}

function isDirectoryMatch(cwd: string, candidate: string): boolean {
  return cwd === candidate || cwd.startsWith(`${candidate}/`);
}

function findModelOverride(
  models: SupermemoryConfigFile["models"] | undefined,
  model: Model<any> | { id?: string; name?: string; provider?: string; api?: string } | undefined,
): { key: string; override: ConfigOverride } | undefined {
  if (!models || !model) return undefined;
  const candidates = modelMatchKeys(model);
  for (const key of candidates) {
    const override = models[key];
    if (override) return { key, override };
  }
  return undefined;
}

function modelMatchKeys(model: Model<any> | { id?: string; name?: string; provider?: string; api?: string }): string[] {
  const id = model.id;
  const name = model.name;
  const provider = model.provider;
  const api = model.api;
  return uniqueStrings([
    id,
    name,
    provider && id ? `${provider}/${id}` : undefined,
    provider && name ? `${provider}/${name}` : undefined,
    provider && id ? `${provider}:${id}` : undefined,
    provider && name ? `${provider}:${name}` : undefined,
    api && id ? `${api}/${id}` : undefined,
  ]);
}

function uniqueStrings(values: Array<string | undefined>): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))];
}

const policyCache = new Map<string, SupermemoryConfigFile | undefined>();

function discoverConfigFiles(cwd: string): string[] {
  const levels: string[][] = [];
  let current = resolve(cwd);
  const homeDir = homedir();

  while (true) {
    const filesAtLevel: string[] = [];

    const supermemoryPath = join(current, ".pi", "supermemory.json");
    const piSupermemoryPath = join(current, ".pi", "pi-supermemory.json");
    const agentPath = join(current, ".pi", "agent", "pi-supermemory.json");

    if (existsSync(supermemoryPath)) filesAtLevel.push(supermemoryPath);
    if (existsSync(piSupermemoryPath)) filesAtLevel.push(piSupermemoryPath);
    if (existsSync(agentPath)) filesAtLevel.push(agentPath);

    if (filesAtLevel.length > 0) {
      levels.push(filesAtLevel);
    }

    if (current === homeDir) break;
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }

  const homeFiles: string[] = [];
  const homeSupermemory = join(homeDir, ".pi", "supermemory.json");
  const homePiSupermemory = join(homeDir, ".pi", "pi-supermemory.json");
  const homeAgent = join(homeDir, ".pi", "agent", "pi-supermemory.json");

  if (existsSync(homeSupermemory)) homeFiles.push(homeSupermemory);
  if (existsSync(homePiSupermemory)) homeFiles.push(homePiSupermemory);
  if (existsSync(homeAgent)) homeFiles.push(homeAgent);

  if (homeFiles.length > 0 && !levels.some((level) => level.some((f) => f === homeSupermemory || f === homePiSupermemory || f === homeAgent))) {
    levels.push(homeFiles);
  }

  return levels.reverse().flat();
}

export function loadMergedPolicyForCwd(cwd: string): SupermemoryConfigFile | undefined {
  const cached = policyCache.get(cwd);
  if (cached !== undefined || policyCache.has(cwd)) return cached;

  const files = discoverConfigFiles(cwd);
  if (files.length === 0) {
    policyCache.set(cwd, undefined);
    return undefined;
  }

  let merged: SupermemoryConfigFile = {};
  for (const file of files) {
    const policy = loadConfigFile(file);
    if (!policy) continue;
    merged = mergePolicies(merged, policy);
  }

  policyCache.set(cwd, merged);
  return merged;
}

function mergePolicies(parent: SupermemoryConfigFile, child: SupermemoryConfigFile): SupermemoryConfigFile {
  const merged: SupermemoryConfigFile = {};
  if (child.default) {
    merged.default = { ...(parent.default ?? {}), ...child.default };
  } else if (parent.default) {
    merged.default = parent.default;
  }
  const directories = mergeDirectoryMaps(parent.directories, child.directories);
  if (directories !== undefined) merged.directories = directories;
  const models = mergeModelMaps(parent.models, child.models);
  if (models !== undefined) merged.models = models;
  const rules = [...(child.rules ?? []), ...(parent.rules ?? [])];
  if (rules.length > 0) merged.rules = rules;
  return merged;
}

function mergeDirectoryMaps(
  parent: SupermemoryConfigFile["directories"],
  child: SupermemoryConfigFile["directories"],
): SupermemoryConfigFile["directories"] {
  if (!child) return parent;
  if (!parent) return child;

  const toRecord = (d: SupermemoryConfigFile["directories"]): Record<string, ConfigOverride> => {
    if (!d) return {};
    if (Array.isArray(d)) {
      return Object.fromEntries(
        d
          .filter((e): e is DirectoryOverride & { path: string } => Boolean(e.path))
          .map((e) => {
            const { path, ...override } = e;
            return [path, override];
          }),
      );
    }
    return d;
  };

  return { ...toRecord(parent), ...toRecord(child) };
}

function mergeModelMaps(
  parent: SupermemoryConfigFile["models"],
  child: SupermemoryConfigFile["models"],
): SupermemoryConfigFile["models"] {
  if (!child) return parent;
  if (!parent) return child;
  return { ...parent, ...child };
}

function findRuleOverride(
  rules: Rule[] | undefined,
  cwd: string | undefined,
  model: Model<any> | { id?: string; name?: string; provider?: string; api?: string } | undefined,
): { path: string; override: ConfigOverride } | undefined {
  if (!rules || !cwd) return undefined;

  const cwdPath = normalize(resolve(cwd));
  const modelKeys = model ? modelMatchKeys(model) : [];

  const matches = rules
    .map((rule, index) => ({ rule, index, pathLength: normalize(resolve(rule.path)).length }))
    .filter(({ rule }) => {
      const rulePath = normalize(resolve(rule.path));
      if (!isDirectoryMatch(cwdPath, rulePath)) return false;
      if (!rule.modelPattern) return true;
      if (modelKeys.length === 0) return false;
      try {
        const regex = new RegExp(rule.modelPattern);
        return modelKeys.some((key) => regex.test(key));
      } catch {
        return false;
      }
    })
    .sort((a, b) => {
      if (b.pathLength !== a.pathLength) return b.pathLength - a.pathLength;
      return a.index - b.index;
    });

  if (matches.length === 0) return undefined;

  const match = matches[0];
  if (!match) return undefined;
  const winner = match.rule;
  const override: ConfigOverride = {};

  if (winner.containerTag !== undefined) override.containerTag = winner.containerTag;
  if (winner.enabled !== undefined) override.enabled = winner.enabled;
  if (winner.apiKey !== undefined) override.apiKey = winner.apiKey;
  if (winner.apiBaseUrl !== undefined) override.apiBaseUrl = winner.apiBaseUrl;
  if (winner.maxRecall !== undefined) override.maxRecall = winner.maxRecall;

  if (winner.permissions !== undefined) {
    switch (winner.permissions) {
      case "read-only":
        override.autoRecall = true;
        override.autoCapture = false;
        break;
      case "write-only":
        override.autoRecall = false;
        override.autoCapture = true;
        break;
      case "read-write":
        override.autoRecall = true;
        override.autoCapture = true;
        break;
    }
    override.permissions = winner.permissions;
  } else {
    if (winner.autoRecall !== undefined) override.autoRecall = winner.autoRecall;
    if (winner.autoCapture !== undefined) override.autoCapture = winner.autoCapture;
  }

  return { path: winner.path, override };
}

function canRead(config: Config | RuntimeConfig): boolean {
  if (!config.enabled) return false;
  if (config.permissions === "write-only") return false;
  return true;
}

function canWrite(config: Config | RuntimeConfig): boolean {
  if (!config.enabled) return false;
  if (config.permissions === "read-only") return false;
  return true;
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
    const nested = candidate.memory ?? candidate.document;
    const content =
      typeof nested === "string"
        ? nested
        : (nested?.content ?? nested?.text ?? candidate.content ?? candidate.text);
    if (!content) return [];
    const id = typeof nested === "string" ? candidate.id : (nested?.id ?? candidate.id);
    const score = candidate.score ?? candidate.similarity;
    const metadata = typeof nested === "string" ? candidate.metadata : (nested?.metadata ?? candidate.metadata);
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
    enabled: config.enabled,
    configured,
    containerTag: config.containerTag,
    apiBaseUrl: config.apiBaseUrl,
    autoRecall: config.autoRecall,
    autoCapture: config.autoCapture,
    permissions: config.permissions,
    maxRecall: config.maxRecall,
    configPath: config.configPath,
    matchedDirectory: config.matchedDirectory,
    matchedModel: config.matchedModel,
    matchedRule: config.matchedRule,
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
