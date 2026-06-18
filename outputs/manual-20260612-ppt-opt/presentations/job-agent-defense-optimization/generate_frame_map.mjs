import fs from "node:fs/promises";
import path from "node:path";

const workspace = path.resolve(
  "/Users/monkey/job-ability-agent-system/outputs/manual-20260612-ppt-opt/presentations/job-agent-defense-optimization",
);
const contentPlan = JSON.parse(
  await fs.readFile(path.join(workspace, "content-plan.json"), "utf8"),
);
const layoutDir = path.join(
  workspace,
  "reference",
  "template-inspect",
  "layouts",
);

const outputSlides = [];
const usedSourceSlides = new Set();

for (const slidePlan of contentPlan.slides) {
  const sourceNumber = String(slidePlan.sourceSlide).padStart(2, "0");
  const layout = JSON.parse(
    await fs.readFile(
      path.join(layoutDir, `source-slide-${sourceNumber}.layout.json`),
      "utf8",
    ),
  );
  const textElements = layout.elements.filter(
    (element) => String(element.textPreview || "").trim().length > 0,
  );

  if (textElements.length !== slidePlan.texts.length) {
    throw new Error(
      `output slide ${slidePlan.outputSlide}: source slide ${slidePlan.sourceSlide} has ${textElements.length} text elements, but content plan has ${slidePlan.texts.length}`,
    );
  }

  usedSourceSlides.add(slidePlan.sourceSlide);
  outputSlides.push({
    outputSlide: slidePlan.outputSlide,
    sourceSlide: slidePlan.sourceSlide,
    narrativeRole: slidePlan.narrativeRole,
    reuseMode: "duplicate-slide",
    editTargets: textElements.map((element, index) => ({
      action: "rewrite",
      sourceElementId: element.aid,
      sourceText: element.textPreview,
      contentIndex: index,
    })),
  });
}

const omittedSourceSlides = [];
for (let sourceSlide = 1; sourceSlide <= 28; sourceSlide += 1) {
  if (!usedSourceSlides.has(sourceSlide)) {
    omittedSourceSlides.push({
      sourceSlide,
      reason: "该页含 EEG 专属图表或不属于本次答辩叙事结构",
    });
  }
}

await fs.writeFile(
  path.join(workspace, "template-frame-map.json"),
  `${JSON.stringify({ outputSlides, omittedSourceSlides }, null, 2)}\n`,
  "utf8",
);

console.log(`generated ${outputSlides.length} output slide mappings`);
