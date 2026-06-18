import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";

import {
  ensureArtifactToolWorkspace,
  importArtifactTool,
  padSlideNumber,
  saveBlobToFile,
} from "/Users/monkey/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs";

const workspace = path.resolve(
  "/Users/monkey/job-ability-agent-system/outputs/manual-20260612-ppt-opt/presentations/job-agent-defense-optimization",
);
const starterPptxPath = path.join(workspace, "template-starter.pptx");
const contentPlanPath = path.join(workspace, "content-plan.json");
const outputDir = path.join(workspace, "output");
const finalPptxPath = path.join(
  outputDir,
  "岗位能力达成学生成长诊断与精准就业智能体系统_优化答辩版.pptx",
);
const previewDir = path.join(workspace, "preview", "final");
const layoutDir = path.join(workspace, "layout", "final");
const contactSheetPath = path.join(workspace, "qa", "final-contact-sheet.png");

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(
  await FileBlob.load(starterPptxPath),
);
const contentPlan = JSON.parse(await fs.readFile(contentPlanPath, "utf8"));

if (presentation.slides.count !== contentPlan.slides.length) {
  throw new Error(
    `starter slide count ${presentation.slides.count} does not match content plan ${contentPlan.slides.length}`,
  );
}

const editLog = [];

for (let slideIndex = 0; slideIndex < presentation.slides.count; slideIndex += 1) {
  const slide = presentation.slides.getItem(slideIndex);
  const slidePlan = contentPlan.slides[slideIndex];
  const textElements = slide.elements.items.filter(
    (element) => element.text && element.text.toString().trim().length > 0,
  );

  if (textElements.length !== slidePlan.texts.length) {
    throw new Error(
      `output slide ${slideIndex + 1}: found ${textElements.length} inherited text elements, expected ${slidePlan.texts.length}`,
    );
  }

  for (let textIndex = 0; textIndex < textElements.length; textIndex += 1) {
    const element = textElements[textIndex];
    const before = element.text.toString();
    const after = slidePlan.texts[textIndex];
    element.text = after;
    editLog.push({
      outputSlide: slideIndex + 1,
      elementId: element.id,
      elementAid: element.aid,
      before,
      after,
    });
  }
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(layoutDir, { recursive: true });
await fs.mkdir(path.dirname(contactSheetPath), { recursive: true });

const previewPaths = [];
for (let slideIndex = 0; slideIndex < presentation.slides.count; slideIndex += 1) {
  const slide = presentation.slides.getItem(slideIndex);
  const padded = padSlideNumber(slideIndex + 1);
  const previewPath = path.join(previewDir, `slide-${padded}.png`);
  const layoutPath = path.join(layoutDir, `slide-${padded}.layout.json`);
  const preview = await presentation.export({ slide, format: "png", scale: 1 });
  const layout = await presentation.export({ slide, format: "layout" });
  await saveBlobToFile(preview, previewPath);
  await saveBlobToFile(layout, layoutPath);
  previewPaths.push(previewPath);
}

const exported = await PresentationFile.exportPptx(presentation);
await exported.save(finalPptxPath);
await fs.writeFile(
  path.join(workspace, "edit-log.json"),
  `${JSON.stringify(editLog, null, 2)}\n`,
  "utf8",
);

const python =
  "/Users/monkey/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const contactScript =
  "/Users/monkey/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/make_contact_sheet.py";
const contact = spawnSync(
  python,
  [contactScript, "--output", contactSheetPath, "--cols", "4", ...previewPaths],
  { encoding: "utf8" },
);
if (contact.status !== 0) {
  throw new Error(contact.stderr || contact.stdout || "contact sheet generation failed");
}

console.log(
  JSON.stringify(
    {
      output: finalPptxPath,
      slides: presentation.slides.count,
      previews: previewDir,
      layouts: layoutDir,
      contactSheet: contactSheetPath,
    },
    null,
    2,
  ),
);
