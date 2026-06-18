from pathlib import Path
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


def set_east_asia_font(run, font_name: str) -> None:
    run.font.name = font_name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font_name)


def remove_run_property(run, property_name: str) -> None:
    r_pr = run._element.get_or_add_rPr()
    element = r_pr.find(qn(property_name))
    if element is not None:
        r_pr.remove(element)


def disable_grid_alignment(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    snap_to_grid = p_pr.find(qn("w:snapToGrid"))
    if snap_to_grid is None:
        snap_to_grid = OxmlElement("w:snapToGrid")
        p_pr.append(snap_to_grid)
    snap_to_grid.set(qn("w:val"), "0")


def main() -> None:
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    document = Document(source)

    target = next(
        (
            paragraph
            for paragraph in document.paragraphs[:20]
            if re.sub(r"\s+", "", paragraph.text) == "技术论文"
        ),
        None,
    )
    if target is None:
        raise RuntimeError("未找到封面的“技术论文”标题段落")

    target.alignment = WD_ALIGN_PARAGRAPH.CENTER
    target.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    target.paragraph_format.line_spacing = Pt(40)
    target.paragraph_format.space_before = Pt(0)
    target.paragraph_format.space_after = Pt(36)
    target.paragraph_format.keep_together = True
    disable_grid_alignment(target)

    for run in target.runs:
        run.font.size = Pt(26)
        run.font.bold = True
        set_east_asia_font(run, "黑体")
        remove_run_property(run, "w:position")
        remove_run_property(run, "w:spacing")

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


if __name__ == "__main__":
    main()
