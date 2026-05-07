import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { DefaultResourceLoader, type ExtensionAPI, type ExtensionCommandContext } from "@mariozechner/pi-coding-agent";
import {
  createSupermemoryExtension,
  resolveSupermemoryConfig,
  loadMergedPolicyForCwd,
  type SupermemoryClient,
  type SupermemorySearchResult,
} from "../src/index.ts";

const NO_CONFIG_PATH = "/tmp/pi-supermemory-test-config-does-not-exist.json";

type RegisteredTool = {
  name: string;
  execute: (
    toolCallId: string,
    params: Record<string, unknown>,
    signal?: AbortSignal,
    onUpdate?: unknown,
    ctx?: Record<string, unknown>,
  ) => Promise<{ content: Array<{ type: "text"; text: string }>; details: unknown }>;
};

type RegisteredCommand = {
  name: string;
  handler: (args: string, ctx: ExtensionCommandContext) => Promise<void>;
};

type EventHandler = (event: Record<string, unknown>, ctx?: Record<string, unknown>) => unknown;

type Harness = {
  tools: Map<string, RegisteredTool>;
  commands: Map<string, RegisteredCommand>;
  handlers: Map<string, EventHandler[]>;
  pi: ExtensionAPI;
};

class FakeSupermemoryClient implements SupermemoryClient {
  searches: Array<{ query: string; limit?: number }> = [];
  saves: Array<{ content: string; isStatic?: boolean; metadata?: Record<string, unknown> }> = [];
  results: SupermemorySearchResult[] = [{ id: "mem_1", content: "Use a shared Supermemory container for dev-agent memory.", score: 0.91 }];

  async search(query: string, options: { limit?: number } = {}): Promise<SupermemorySearchResult[]> {
    this.searches.push({ query, ...(options.limit === undefined ? {} : { limit: options.limit }) });
    return this.results;
  }

  async save(content: string, options: { isStatic?: boolean; metadata?: Record<string, unknown> } = {}) {
    this.saves.push({
      content,
      ...(options.isStatic === undefined ? {} : { isStatic: options.isStatic }),
      ...(options.metadata === undefined ? {} : { metadata: options.metadata }),
    });
    return { ok: true, status: 200, id: "mem_saved" };
  }
}

test("registers Supermemory tools and command", () => {
  const harness = createHarness();
  createSupermemoryExtension({ client: new FakeSupermemoryClient(), configPath: NO_CONFIG_PATH, clock: () => 1 }).register(harness.pi);

  assert.ok(harness.tools.has("supermemory_search"));
  assert.ok(harness.tools.has("supermemory_save"));
  assert.ok(harness.tools.has("supermemory_status"));
  assert.ok(harness.commands.has("supermemory"));
  assert.ok(harness.handlers.has("input"));
  assert.ok(harness.handlers.has("context"));
  assert.ok(harness.handlers.has("turn_end"));
});

test("search tool queries Supermemory with bounded limit", async () => {
  const client = new FakeSupermemoryClient();
  const harness = createHarness();
  createSupermemoryExtension({ client, configPath: NO_CONFIG_PATH, maxRecall: 3, clock: () => 1 }).register(harness.pi);

  const tool = requireTool(harness, "supermemory_search");
  const result = await tool.execute("call_1", { query: "memory config", limit: 99 });

  assert.deepEqual(client.searches, [{ query: "memory config", limit: 25 }]);
  assert.match(JSON.stringify(result.details), /shared Supermemory container/);
});

test("save tool writes into Supermemory with source metadata", async () => {
  const client = new FakeSupermemoryClient();
  const harness = createHarness();
  createSupermemoryExtension({ client, configPath: NO_CONFIG_PATH, containerTag: "team-dev-memory", clock: () => 1 }).register(harness.pi);

  const tool = requireTool(harness, "supermemory_save");
  const result = await tool.execute("call_1", { content: "Remember this", is_static: true });

  assert.equal(result.details && typeof result.details === "object" && "id" in result.details ? result.details.id : undefined, "mem_saved");
  assert.equal(client.saves.length, 1);
  assert.equal(client.saves[0]?.content, "Remember this");
  assert.equal(client.saves[0]?.isStatic, true);
  assert.equal(client.saves[0]?.metadata?.source, "pi-supermemory");
  assert.equal(client.saves[0]?.metadata?.container_tag, "team-dev-memory");
});

test("context hook injects recall results before existing messages", async () => {
  const client = new FakeSupermemoryClient();
  const harness = createHarness();
  createSupermemoryExtension({ client, configPath: NO_CONFIG_PATH, containerTag: "pi-supermemory", clock: () => 123 }).register(harness.pi);

  await emit(harness, "input", { source: "user", content: "How should dev memory be scoped?" });
  const result = (await emit(harness, "context", {
    messages: [{ role: "user", content: "Original prompt", timestamp: 100 }],
  })) as { messages: Array<{ role: string; content: string; timestamp: number }> };

  assert.equal(client.searches[0]?.query, "How should dev memory be scoped?");
  assert.equal(result.messages.length, 2);
  assert.equal(result.messages[0]?.role, "user");
  assert.match(result.messages[0]?.content ?? "", /Relevant Supermemory context/);
  assert.match(result.messages[0]?.content ?? "", /pi-supermemory/);
  assert.equal(result.messages[0]?.timestamp, 123);
});

test("turn_end hook captures completed user and assistant turns", async () => {
  const client = new FakeSupermemoryClient();
  const harness = createHarness();
  createSupermemoryExtension({ client, configPath: NO_CONFIG_PATH, clock: () => 1 }).register(harness.pi);

  await emit(harness, "input", { source: "user", content: "Wire Pi to Supermemory" });
  await emit(harness, "turn_end", {
    message: {
      role: "assistant",
      content: [{ type: "text", text: "Pi now uses Supermemory via direct API client." }],
      timestamp: 2,
    },
  });

  assert.equal(client.saves.length, 1);
  assert.match(client.saves[0]?.content ?? "", /Wire Pi to Supermemory/);
  assert.match(client.saves[0]?.content ?? "", /direct API client/);
  assert.equal(client.saves[0]?.metadata?.capture_mode, "turn_end");
});

test("status tool reports missing API key without throwing", async () => {
  const harness = createHarness();
  createSupermemoryExtension({ apiKey: "", configPath: NO_CONFIG_PATH, containerTag: "pi-supermemory", clock: () => 1 }).register(harness.pi);

  const tool = requireTool(harness, "supermemory_status");
  const result = await tool.execute("call_1", {});

  assert.match(JSON.stringify(result.details), /"configured":false/);
  assert.match(JSON.stringify(result.details), /pi-supermemory/);
});

test("config resolver applies default, directory, and model precedence", () => {
  const result = resolveSupermemoryConfig({
    base: { containerTag: "base-memory", maxRecall: 3 },
    cwd: "/workspace/app/packages/api",
    model: { id: "gpt-5.5", name: "GPT 5.5", provider: "openai-codex", api: "openai-codex-responses" },
    policy: {
      default: { containerTag: "default-memory", maxRecall: 4 },
      directories: {
        "/workspace": { containerTag: "workspace-memory", maxRecall: 5 },
        "/workspace/app": { containerTag: "app-memory", maxRecall: 6 },
      },
      models: {
        "openai-codex/gpt-5.5": { containerTag: "model-memory", maxRecall: 7 },
      },
    },
  });

  assert.equal(result.containerTag, "model-memory");
  assert.equal(result.maxRecall, 7);
  assert.equal(result.matchedDirectory, "/workspace/app");
  assert.equal(result.matchedModel, "openai-codex/gpt-5.5");
});

test("config resolver lets directory override default when model does not match", () => {
  const result = resolveSupermemoryConfig({
    base: { containerTag: "base-memory" },
    cwd: "/workspace/app",
    model: { id: "other-model", name: "Other Model", provider: "openai-codex", api: "openai-codex-responses" },
    policy: {
      default: { containerTag: "default-memory" },
      directories: {
        "/workspace/app": { containerTag: "app-memory" },
      },
      models: {
        "openai-codex/gpt-5.5": { containerTag: "model-memory" },
      },
    },
  });

  assert.equal(result.containerTag, "app-memory");
  assert.equal(result.matchedDirectory, "/workspace/app");
  assert.equal(result.matchedModel, undefined);
});

test("config resolver supports model disable override over directory enable", () => {
  const result = resolveSupermemoryConfig({
    base: { enabled: true, containerTag: "base-memory" },
    cwd: "/workspace/app",
    model: { id: "no-memory-model", name: "No Memory Model", provider: "local", api: "openai-responses" },
    policy: {
      default: { enabled: true, containerTag: "default-memory" },
      directories: {
        "/workspace/app": { enabled: true, containerTag: "app-memory" },
      },
      models: {
        "local/no-memory-model": { enabled: false },
      },
    },
  });

  assert.equal(result.enabled, false);
  assert.equal(result.containerTag, "app-memory");
  assert.equal(result.matchedModel, "local/no-memory-model");
});

test("disabled model config prevents search even when a client exists", async () => {
  const client = new FakeSupermemoryClient();
  const harness = createHarness();
  const configDir = await mkdtemp(join(tmpdir(), "pi-supermemory-config-"));
  const configPath = join(configDir, "pi-supermemory.json");
  await writeFile(
    configPath,
    JSON.stringify({
      default: { containerTag: "default-memory" },
      directories: { "/workspace/app": { containerTag: "app-memory" } },
      models: { "local/no-memory-model": { enabled: false } },
    }),
  );
  createSupermemoryExtension({
    client,
    configPath,
    clock: () => 1,
  }).register(harness.pi);

  try {
    const tool = requireTool(harness, "supermemory_search");
    const result = await tool.execute("call_1", { query: "memory config" }, undefined, undefined, {
      cwd: "/workspace/app",
      model: { id: "no-memory-model", name: "No Memory Model", provider: "local", api: "openai-responses" },
    });

    assert.match(JSON.stringify(result.details), /disabled by configuration/);
    assert.equal(client.searches.length, 0);
  } finally {
    await rm(configDir, { recursive: true, force: true });
  }
});

test("Pi SDK discovers the project-local pi-supermemory extension", async () => {
  const agentDir = await mkdtemp(join(tmpdir(), "pi-supermemory-e2e-"));
  try {
    const loader = new DefaultResourceLoader({
      cwd: process.cwd(),
      agentDir,
      noSkills: true,
      noPromptTemplates: true,
      noThemes: true,
      noContextFiles: true,
    });
    await loader.reload();

    const extensions = loader.getExtensions();
    assert.deepEqual(extensions.errors, []);
    assert.ok(
      extensions.extensions.some((extension) => extension.resolvedPath.endsWith(".pi/extensions/pi-supermemory/index.ts")),
      "expected DefaultResourceLoader to discover .pi/extensions/pi-supermemory/index.ts",
    );
  } finally {
    await rm(agentDir, { recursive: true, force: true });
  }
});

test("hierarchical config discovery merges parent and child configs", async () => {
  const rootDir = await mkdtemp(join(tmpdir(), "pi-supermemory-hier-"));
  const parentDir = join(rootDir, "parent");
  const childDir = join(parentDir, "child");
  await mkdir(join(parentDir, ".pi", "agent"), { recursive: true });
  await mkdir(join(childDir, ".pi", "agent"), { recursive: true });

  await writeFile(
    join(parentDir, ".pi", "agent", "pi-supermemory.json"),
    JSON.stringify({ default: { containerTag: "parent-memory", maxRecall: 5 } }),
  );
  await writeFile(
    join(childDir, ".pi", "agent", "pi-supermemory.json"),
    JSON.stringify({ default: { maxRecall: 10 } }),
  );

  try {
    const policy = loadMergedPolicyForCwd(childDir);
    const result = policy ? resolveSupermemoryConfig({ cwd: childDir, policy }) : resolveSupermemoryConfig({ cwd: childDir });
    assert.equal(result.containerTag, "parent-memory");
    assert.equal(result.maxRecall, 10);
  } finally {
    await rm(rootDir, { recursive: true, force: true });
  }
});

test("hierarchical config prefers child directory overrides over parent", async () => {
  const rootDir = await mkdtemp(join(tmpdir(), "pi-supermemory-hier-"));
  const parentDir = join(rootDir, "workspace");
  const childDir = join(parentDir, "app");
  await mkdir(join(parentDir, ".pi", "agent"), { recursive: true });
  await mkdir(join(childDir, ".pi", "agent"), { recursive: true });

  await writeFile(
    join(parentDir, ".pi", "agent", "pi-supermemory.json"),
    JSON.stringify({ directories: { [parentDir]: { containerTag: "workspace-memory" } } }),
  );
  await writeFile(
    join(childDir, ".pi", "agent", "pi-supermemory.json"),
    JSON.stringify({ directories: { [childDir]: { containerTag: "app-memory" } } }),
  );

  try {
    const policy = loadMergedPolicyForCwd(childDir);
    const result = policy ? resolveSupermemoryConfig({ cwd: childDir, policy }) : resolveSupermemoryConfig({ cwd: childDir });
    assert.equal(result.containerTag, "app-memory");
    assert.equal(result.matchedDirectory, childDir);
  } finally {
    await rm(rootDir, { recursive: true, force: true });
  }
});

test("multiple config files at same level merge with specificity priority", async () => {
  const rootDir = await mkdtemp(join(tmpdir(), "pi-supermemory-multi-"));
  const projectDir = join(rootDir, "project");
  await mkdir(join(projectDir, ".pi"), { recursive: true });

  await writeFile(
    join(projectDir, ".pi", "supermemory.json"),
    JSON.stringify({ default: { containerTag: "generic-memory" } }),
  );
  await writeFile(
    join(projectDir, ".pi", "pi-supermemory.json"),
    JSON.stringify({ default: { containerTag: "specific-memory" } }),
  );

  try {
    const policy = loadMergedPolicyForCwd(projectDir);
    const result = policy ? resolveSupermemoryConfig({ cwd: projectDir, policy }) : resolveSupermemoryConfig({ cwd: projectDir });
    assert.equal(result.containerTag, "specific-memory");
  } finally {
    await rm(rootDir, { recursive: true, force: true });
  }
});

test("rules match provider/model regex and apply permissions", () => {
  const result = resolveSupermemoryConfig({
    base: { containerTag: "base-memory" },
    cwd: "/workspace/app",
    model: { id: "gpt-5.5", name: "GPT 5.5", provider: "openai-codex", api: "openai-codex-responses" },
    policy: {
      rules: [
        { path: "/workspace", modelPattern: "anthropic/.*", containerTag: "anthropic-memory", permissions: "read-only" },
        { path: "/workspace/app", modelPattern: "openai-codex/.*", containerTag: "codex-memory", permissions: "read-only" },
      ],
    },
  });

  assert.equal(result.containerTag, "codex-memory");
  assert.equal(result.permissions, "read-only");
  assert.equal(result.autoRecall, true);
  assert.equal(result.autoCapture, false);
  assert.equal(result.matchedRule, "/workspace/app [read-only]");
});

test("rules fall back to less specific path when model does not match", () => {
  const result = resolveSupermemoryConfig({
    base: { containerTag: "base-memory" },
    cwd: "/workspace/app",
    model: { id: "claude-4", name: "Claude 4", provider: "anthropic", api: "anthropic-responses" },
    policy: {
      rules: [
        { path: "/workspace", containerTag: "workspace-memory" },
        { path: "/workspace/app", modelPattern: "openai-codex/.*", containerTag: "codex-memory" },
      ],
    },
  });

  assert.equal(result.containerTag, "workspace-memory");
  assert.equal(result.matchedRule, "/workspace");
});

test("read-only permission blocks save tool", async () => {
  const client = new FakeSupermemoryClient();
  const harness = createHarness();
  const configDir = await mkdtemp(join(tmpdir(), "pi-supermemory-config-"));
  const configPath = join(configDir, "pi-supermemory.json");
  await writeFile(
    configPath,
    JSON.stringify({
      rules: [{ path: "/workspace/app", modelPattern: "openai-codex/.*", permissions: "read-only", containerTag: "read-only-memory" }],
    }),
  );
  createSupermemoryExtension({ client, configPath, clock: () => 1 }).register(harness.pi);

  try {
    const tool = requireTool(harness, "supermemory_save");
    const result = await tool.execute("call_1", { content: "test" }, undefined, undefined, {
      cwd: "/workspace/app",
      model: { id: "gpt-5.5", name: "GPT 5.5", provider: "openai-codex", api: "openai-codex-responses" },
    });

    assert.match(JSON.stringify(result.details), /save is disabled by configuration \(read-only\)/);
    assert.equal(client.saves.length, 0);
  } finally {
    await rm(configDir, { recursive: true, force: true });
  }
});

test("write-only permission blocks search tool", async () => {
  const client = new FakeSupermemoryClient();
  const harness = createHarness();
  const configDir = await mkdtemp(join(tmpdir(), "pi-supermemory-config-"));
  const configPath = join(configDir, "pi-supermemory.json");
  await writeFile(
    configPath,
    JSON.stringify({
      rules: [{ path: "/workspace/app", modelPattern: "openai-codex/.*", permissions: "write-only", containerTag: "write-only-memory" }],
    }),
  );
  createSupermemoryExtension({ client, configPath, clock: () => 1 }).register(harness.pi);

  try {
    const tool = requireTool(harness, "supermemory_search");
    const result = await tool.execute("call_1", { query: "test" }, undefined, undefined, {
      cwd: "/workspace/app",
      model: { id: "gpt-5.5", name: "GPT 5.5", provider: "openai-codex", api: "openai-codex-responses" },
    });

    assert.match(JSON.stringify(result.details), /search is disabled by configuration \(write-only\)/);
    assert.equal(client.searches.length, 0);
  } finally {
    await rm(configDir, { recursive: true, force: true });
  }
});

function createHarness(): Harness {
  const tools = new Map<string, RegisteredTool>();
  const commands = new Map<string, RegisteredCommand>();
  const handlers = new Map<string, EventHandler[]>();
  const pi = {
    registerTool(tool: RegisteredTool) {
      tools.set(tool.name, tool);
    },
    registerCommand(name: string, command: Omit<RegisteredCommand, "name">) {
      commands.set(name, { name, ...command });
    },
    on(eventName: string, handler: EventHandler) {
      const existing = handlers.get(eventName) ?? [];
      existing.push(handler);
      handlers.set(eventName, existing);
    },
  } as unknown as ExtensionAPI;

  return { tools, commands, handlers, pi };
}

function requireTool(harness: Harness, name: string): RegisteredTool {
  const tool = harness.tools.get(name);
  assert.ok(tool, `expected tool ${name} to be registered`);
  return tool;
}

async function emit(harness: Harness, eventName: string, event: Record<string, unknown>, ctx?: Record<string, unknown>): Promise<unknown> {
  let result: unknown;
  for (const handler of harness.handlers.get(eventName) ?? []) {
    result = await handler(event, ctx);
  }
  return result;
}
