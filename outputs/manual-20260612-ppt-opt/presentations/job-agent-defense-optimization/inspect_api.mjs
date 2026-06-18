import path from "node:path";

import {
  ensureArtifactToolWorkspace,
  importArtifactTool,
} from "/Users/monkey/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs";

const workspace = path.resolve(
  "/Users/monkey/job-ability-agent-system/outputs/manual-20260612-ppt-opt/presentations/job-agent-defense-optimization",
);
const source = "/Users/monkey/Desktop/飞腾杯/基于领域对抗多头残差图卷积网络的EEG情绪识别模型优化与端侧部署.pptx";

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const slide = presentation.slides.getItem(0);

console.log("presentation keys", Object.keys(presentation));
console.log("slides prototype", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides)));
console.log("slide keys", Object.keys(slide));
console.log("slide prototype", Object.getOwnPropertyNames(Object.getPrototypeOf(slide)));
console.log("elements", slide.elements?.items?.length, Object.getOwnPropertyNames(Object.getPrototypeOf(slide.elements)));

for (let index = 0; index < Math.min(slide.elements.items.length, 12); index += 1) {
  const element = slide.elements.items[index];
  console.log("element", index, {
    keys: Object.keys(element),
    proto: Object.getOwnPropertyNames(Object.getPrototypeOf(element)),
    id: element.id,
    aid: element.aid,
    name: element.name,
    type: element.type,
    text: String(element.text || "").slice(0, 120),
  });
  if (element.text) {
    console.log("text object", {
      textType: typeof element.text,
      textKeys: Object.keys(element.text),
      textProto: Object.getOwnPropertyNames(Object.getPrototypeOf(element.text)),
      value: element.text.toString(),
      fontSize: element.text.fontSize,
      typeface: element.text.typeface,
      color: element.text.color,
    });
  }
}
