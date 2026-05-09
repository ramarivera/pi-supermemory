import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createSupermemoryExtension } from "../../../src/index.ts";

function localPiSupermemoryExtension(pi: ExtensionAPI): void {
  createSupermemoryExtension({ commandName: "local-supermemory", toolNamePrefix: "local_" }).register(pi);
}

export * from "../../../src/index.ts";
export default localPiSupermemoryExtension;
