import fs from "node:fs/promises";
import path from "node:path";

const workspace = path.resolve(
  "/Users/monkey/job-ability-agent-system/outputs/manual-20260612-ppt-opt/presentations/job-agent-defense-optimization",
);
const sourcePath = path.join(
  workspace,
  "reference",
  "template-inspect",
  "template-inspect.ndjson",
);
const outputPath = path.join(workspace, "template-inspect-normalized.ndjson");

const lines = (await fs.readFile(sourcePath, "utf8"))
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => {
    const record = JSON.parse(line);
    if (Number.isInteger(record.slide)) {
      record.slide -= 1;
    }
    return JSON.stringify(record);
  });

await fs.writeFile(outputPath, `${lines.join("\n")}\n`, "utf8");
console.log(outputPath);
