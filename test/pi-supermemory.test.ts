import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { DefaultResourceLoader, type ExtensionAPI, type ExtensionCommandContext } from "@mariozechner/pi-coding-agent";
import { createSupermemoryExtension, type SupermemoryClient, type SupermemorySearchResult } from "../src/index.ts";

type RegisteredTool = {
  name: string;
  execute: (toolCallId: string, params: Record<string, unknown>) => Promise<{ content: Array<{ type: "text"; text: string }>; details: unknown }>;
};

type RegisteredCommand = {
  name: string;
  handler: (args: string, ctx: ExtensionCommandContext) => Promise<void>;
};

type EventHandler = (event: Record<string, unknown>) => unknown;

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
  createSupermemoryExtension({ client: new FakeSupermemoryClient(), clock: () => 1 }).register(harness.pi);

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
  createSupermemoryExtension({ client, maxRecall: 3, clock: () => 1 }).register(harness.pi);

  const tool = requireTool(harness, "supermemory_search");
  const result = await tool.execute("call_1", { query: "memory config", limit: 99 });

  assert.deepEqual(client.searches, [{ query: "memory config", limit: 25 }]);
  assert.match(JSON.stringify(result.details), /shared Supermemory container/);
});

test("save tool writes into Supermemory with source metadata", async () => {
  const client = new FakeSupermemoryClient();
  const harness = createHarness();
  createSupermemoryExtension({ client, containerTag: "team-dev-memory", clock: () => 1 }).register(harness.pi);

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
  createSupermemoryExtension({ client, containerTag: "pi-supermemory", clock: () => 123 }).register(harness.pi);

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
  createSupermemoryExtension({ client, clock: () => 1 }).register(harness.pi);

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
  createSupermemoryExtension({ apiKey: "", containerTag: "pi-supermemory", clock: () => 1 }).register(harness.pi);

  const tool = requireTool(harness, "supermemory_status");
  const result = await tool.execute("call_1", {});

  assert.match(JSON.stringify(result.details), /"configured":false/);
  assert.match(JSON.stringify(result.details), /pi-supermemory/);
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

async function emit(harness: Harness, eventName: string, event: Record<string, unknown>): Promise<unknown> {
  let result: unknown;
  for (const handler of harness.handlers.get(eventName) ?? []) {
    result = await handler(event);
  }
  return result;
}
