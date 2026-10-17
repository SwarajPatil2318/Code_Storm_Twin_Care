import ts from "typescript";
import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
const source = await readFile(
  new URL("./src/sourceTiming.ts", import.meta.url),
  "utf8",
);
const js = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { sourceReceipt } = await import(
  "data:text/javascript;base64," + Buffer.from(js).toString("base64")
);
const cutoff = "2025-01-08T12:00:00Z";
const row = (received, observed = "2025-01-08T09:00:00Z", value = 72) => ({
  concept: "heart_rate",
  available_time: received,
  observed_time: observed,
  value: { usable_value: value },
});
assert.equal(
  sourceReceipt([row("2025-01-08T10:00:00Z")], ["heart_rate"], cutoff).label,
  "Synced 2h ago",
);
const earlier = row("2025-01-08T06:00:00Z");
assert.equal(
  sourceReceipt(
    [
      earlier,
      row("2025-01-08T13:00:00Z"),
      row("2025-01-08T11:00:00Z", "2025-01-08T13:00:00Z"),
    ],
    ["heart_rate"],
    cutoff,
  ).label,
  "Synced 6h ago",
);
assert.equal(
  sourceReceipt(
    [earlier, row("2025-01-08T11:00:00Z", undefined, null)],
    ["heart_rate"],
    cutoff,
  ).label,
  "Synced 6h ago",
);
assert.equal(
  sourceReceipt([row("2025-01-08T11:30:00Z")], ["heart_rate"], cutoff).label,
  "Synced <1h ago",
);
assert.equal(
  sourceReceipt([row("2025-01-06T12:00:00Z")], ["heart_rate"], cutoff).label,
  "Synced 2d ago",
);
assert.equal(sourceReceipt([earlier], ["spo2"], cutoff).available, false);
assert.equal(
  sourceReceipt([row(null)], ["heart_rate"], cutoff).available,
  false,
);
assert.equal(
  sourceReceipt(
    [row("2025-01-08T10:00:00Z")],
    ["heart_rate"],
    "2025-01-08T09:00:00Z",
  ).available,
  false,
);
console.log(
  "8 source-timing checks passed: future receipt/observation exclusion, missingness, units of age, cutoff and device isolation.",
);
