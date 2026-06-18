from __future__ import annotations

import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 24
DURATION = 55.0
FRAME_COUNT = int(FPS * DURATION)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "java_resume_job_matching_silent.mp4"
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"

BG_TOP = (242, 248, 255)
BG_BOTTOM = (224, 238, 255)
INK = (18, 35, 62)
MUTED = (85, 105, 133)
BLUE = (48, 101, 226)
CYAN = (35, 180, 218)
GREEN = (49, 178, 116)
ORANGE = (244, 147, 55)
RED = (224, 79, 92)
WHITE = (255, 255, 255)
PALE_BLUE = (232, 241, 255)
PALE_GREEN = (229, 248, 238)
PALE_ORANGE = (255, 242, 224)
LINE = (196, 215, 241)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


FONTS = {
    "hero": font(54),
    "h1": font(40),
    "h2": font(30),
    "body": font(23),
    "small": font(18),
    "tiny": font(15),
    "number": font(62),
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease(value: float) -> float:
    value = clamp(value)
    return 1 - (1 - value) ** 3


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3 - 2 * value)


def scene_progress(t: float, start: float, end: float) -> float:
    return clamp((t - start) / (end - start))


def background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_TOP)
    pixels = image.load()
    for y in range(HEIGHT):
        ratio = y / (HEIGHT - 1)
        color = tuple(
            int(BG_TOP[i] * (1 - ratio) + BG_BOTTOM[i] * ratio)
            for i in range(3)
        )
        for x in range(WIDTH):
            pixels[x, y] = color

    draw = ImageDraw.Draw(image, "RGBA")
    for x in range(0, WIDTH, 80):
        draw.line((x, 0, x, HEIGHT), fill=(80, 130, 200, 18), width=1)
    for y in range(0, HEIGHT, 80):
        draw.line((0, y, WIDTH, y), fill=(80, 130, 200, 18), width=1)
    draw.ellipse((960, -170, 1350, 220), fill=(70, 135, 255, 18))
    draw.ellipse((-180, 500, 220, 900), fill=(24, 196, 190, 16))
    return image


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill,
    outline=None,
    radius: int = 22,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def centered(draw: ImageDraw.ImageDraw, xy, text: str, text_font, fill=INK) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        text,
        font=text_font,
        fill=fill,
    )


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    text_font,
    fill=INK,
    spacing: int = 6,
    align: str = "left",
) -> None:
    x1, y1, x2, _ = box
    max_width = x2 - x1
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        width = draw.textbbox((0, 0), candidate, font=text_font)[2]
        if width > max_width and current:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)

    y = y1
    line_height = draw.textbbox((0, 0), "示例", font=text_font)[3] + spacing
    for line in lines:
        if align == "center":
            line_width = draw.textbbox((0, 0), line, font=text_font)[2]
            x = x1 + (max_width - line_width) / 2
        else:
            x = x1
        draw.text((x, y), line, font=text_font, fill=fill)
        y += line_height


def arrow(draw: ImageDraw.ImageDraw, start, end, color=BLUE, width=5) -> None:
    draw.line((*start, *end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 15
    for delta in (2.55, -2.55):
        point = (
            end[0] + length * math.cos(angle + delta),
            end[1] + length * math.sin(angle + delta),
        )
        draw.line((*end, *point), fill=color, width=width)


def chip(draw, x, y, text, color=BLUE, alpha=255, size="small") -> int:
    chip_font = FONTS[size]
    text_box = draw.textbbox((0, 0), text, font=chip_font)
    width = text_box[2] - text_box[0] + 30
    height = text_box[3] - text_box[1] + 18
    rounded(
        draw,
        (int(x), int(y), int(x + width), int(y + height)),
        (*color, min(alpha, 35)) if len(color) == 3 else color,
        (*color, alpha) if len(color) == 3 else color,
        radius=height // 2,
        width=2,
    )
    draw.text(
        (x + 15, y + 6),
        text,
        font=chip_font,
        fill=(*color, alpha) if len(color) == 3 else color,
    )
    return width


def header(draw, active: int, title: str, subtitle: str) -> None:
    draw.text((60, 35), title, font=FONTS["h1"], fill=INK)
    draw.text((62, 88), subtitle, font=FONTS["small"], fill=MUTED)
    labels = ["简历输入", "画像提取", "向量召回", "规则筛选", "AI精排", "TOP5"]
    x0 = 675
    for index, label in enumerate(labels):
        x = x0 + index * 92
        color = BLUE if index <= active else (163, 181, 207)
        draw.ellipse((x, 49, x + 18, 67), fill=color)
        if index < len(labels) - 1:
            draw.line((x + 19, 58, x + 82, 58), fill=color, width=3)
        centered(draw, (x + 9, 82), label, FONTS["tiny"], color)


def resume_card(draw, x: float, y: float, scale: float = 1.0, alpha: int = 255) -> None:
    w, h = int(390 * scale), int(500 * scale)
    rounded(
        draw,
        (int(x), int(y), int(x + w), int(y + h)),
        (*WHITE, alpha),
        (*LINE, alpha),
        radius=int(28 * scale),
        width=max(1, int(2 * scale)),
    )
    draw.rectangle(
        (int(x), int(y), int(x + w), int(y + 12 * scale)),
        fill=(*BLUE, alpha),
    )
    avatar_r = int(42 * scale)
    avatar_x = int(x + 65 * scale)
    avatar_y = int(y + 80 * scale)
    draw.ellipse(
        (
            avatar_x - avatar_r,
            avatar_y - avatar_r,
            avatar_x + avatar_r,
            avatar_y + avatar_r,
        ),
        fill=(*PALE_BLUE, alpha),
        outline=(*BLUE, alpha),
        width=max(1, int(3 * scale)),
    )
    centered(draw, (avatar_x, avatar_y), "黎", font(max(18, int(32 * scale))), (*BLUE, alpha))
    draw.text(
        (x + 125 * scale, y + 48 * scale),
        "黎盛",
        font=font(max(18, int(30 * scale))),
        fill=(*INK, alpha),
    )
    draw.text(
        (x + 125 * scale, y + 92 * scale),
        "Java 后端开发方向",
        font=font(max(13, int(18 * scale))),
        fill=(*MUTED, alpha),
    )
    sections = [
        ("教育背景", "地理空间信息工程 · 本科"),
        ("核心技能", "Java / Spring Boot / MySQL / Redis"),
        ("微服务", "Spring Cloud / Nacos / RabbitMQ"),
        ("项目经历", "物流运输平台 · 校园 App · GIS"),
    ]
    yy = y + 150 * scale
    for label, value in sections:
        draw.text(
            (x + 30 * scale, yy),
            label,
            font=font(max(12, int(16 * scale))),
            fill=(*BLUE, alpha),
        )
        draw.text(
            (x + 30 * scale, yy + 28 * scale),
            value,
            font=font(max(12, int(17 * scale))),
            fill=(*INK, alpha),
        )
        draw.line(
            (x + 30 * scale, yy + 60 * scale, x + w - 30 * scale, yy + 60 * scale),
            fill=(*LINE, alpha),
            width=1,
        )
        yy += 76 * scale


def portal(draw, x: float, y: float, pulse: float) -> None:
    for radius, opacity in [(82, 25), (64, 45), (48, 80)]:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(*CYAN, int(opacity * pulse)),
            width=5,
        )
    rounded(
        draw,
        (int(x - 38), int(y - 45), int(x + 38), int(y + 45)),
        (*BLUE, 235),
        (*WHITE, 220),
        radius=16,
        width=2,
    )
    centered(draw, (x, y), "AI", FONTS["h2"], WHITE)


def draw_scene_1(draw, t: float) -> None:
    p = scene_progress(t, 0, 7)
    header(draw, 0, "从一份简历，到精准岗位 TOP5", "真实简历 · 679条岗位 · 全流程动画演示")
    card_x = -420 + ease(clamp(p / 0.35)) * 510
    resume_card(draw, card_x, 155, 0.9)
    portal(draw, 1040, 385, 0.65 + 0.35 * math.sin(t * 4) ** 2)

    arrow_alpha = int(255 * clamp((p - 0.18) / 0.22))
    arrow(draw, (520, 385), (900, 385), (*BLUE, arrow_alpha), 7)
    centered(draw, (710, 340), "上传并解析", FONTS["h2"], (*BLUE, arrow_alpha))
    centered(draw, (710, 435), "PDF · 3页", FONTS["body"], (*MUTED, arrow_alpha))
    if p > 0.62:
        travel = ease((p - 0.62) / 0.38)
        x = 520 + 360 * travel
        rounded(draw, (int(x), 360, int(x + 90), 410), (*WHITE, 245), (*BLUE, 255), 12, 2)
        centered(draw, (x + 45, 385), "简历", FONTS["small"], BLUE)


def draw_scene_2(draw, t: float) -> None:
    p = scene_progress(t, 7, 14)
    header(draw, 1, "第一步：提取学生能力画像", "把非结构化简历转换为可计算字段")
    resume_card(draw, 70, 170, 0.72)
    arrow(draw, (390, 380), (510, 380), BLUE, 6)
    rounded(draw, (530, 150, 1200, 625), WHITE, LINE, 28, 2)
    draw.text((570, 185), "学生检索画像", font=FONTS["h2"], fill=INK)
    fields = [
        ("目标岗位", "Java后端工程师", BLUE),
        ("专业学历", "地理空间信息工程 · 本科", CYAN),
        ("核心技能", "Java · Spring Boot · MySQL · Redis", GREEN),
        ("微服务", "Spring Cloud · Nacos · RabbitMQ", ORANGE),
        ("项目证据", "物流运输平台 · 校园App · GIS", BLUE),
        ("工具部署", "Docker · Linux · Nginx · Git", GREEN),
    ]
    for i, (label, value, color) in enumerate(fields):
        appear = ease(clamp((p - i * 0.09) / 0.25))
        y = 245 + i * 57
        x = 570 + (1 - appear) * 160
        rounded(
            draw,
            (int(x), y, 1155, y + 43),
            (*color, int(24 * appear)),
            (*color, int(220 * appear)),
            13,
            2,
        )
        draw.text((x + 16, y + 10), label, font=FONTS["small"], fill=(*color, int(255 * appear)))
        draw.text((x + 145, y + 10), value, font=FONTS["small"], fill=(*INK, int(255 * appear)))
    centered(draw, (865, 590), "形成学生稀疏向量 S", FONTS["body"], BLUE)


def draw_database(draw, x: int, y: int, count: int, alpha: int = 255) -> None:
    draw.rectangle((x, y + 30, x + 250, y + 210), fill=(*PALE_BLUE, alpha), outline=(*BLUE, alpha), width=3)
    draw.ellipse((x, y, x + 250, y + 65), fill=(*WHITE, alpha), outline=(*BLUE, alpha), width=3)
    draw.ellipse((x, y + 175, x + 250, y + 240), fill=(*PALE_BLUE, alpha), outline=(*BLUE, alpha), width=3)
    centered(draw, (x + 125, y + 93), str(count), FONTS["number"], (*BLUE, alpha))
    centered(draw, (x + 125, y + 145), "条岗位记录", FONTS["body"], (*INK, alpha))


def draw_scene_3(draw, t: float) -> None:
    p = scene_progress(t, 14, 23)
    header(draw, 2, "第二步：679条岗位进行向量召回", "学生向量与每个岗位向量计算余弦相似度")
    draw_database(draw, 80, 220, 679)
    rounded(draw, (430, 185, 785, 600), WHITE, LINE, 26, 2)
    centered(draw, (607, 225), "相似度计算", FONTS["h2"], INK)
    centered(draw, (607, 290), "cos(S, J)", FONTS["hero"], BLUE)
    centered(draw, (607, 350), "文本共同词越多，方向越接近", FONTS["small"], MUTED)
    shared = ["java", "spring", "mysql", "redis", "后端", "微服务"]
    x = 472
    y = 395
    for i, label in enumerate(shared):
        if i == 3:
            x = 500
            y += 58
        x += chip(draw, x, y, label, (BLUE, CYAN, GREEN)[i % 3]) + 12
    draw.text((485, 520), "逐岗计算 → 相似度降序排序", font=FONTS["body"], fill=INK)
    arrow(draw, (350, 345), (420, 345), BLUE, 6)
    arrow(draw, (800, 345), (885, 345), BLUE, 6)

    count_p = ease(clamp((p - 0.42) / 0.45))
    current = round(679 - (679 - 300) * count_p)
    rounded(draw, (900, 210, 1190, 520), WHITE, LINE, 26, 2)
    centered(draw, (1045, 260), "向量召回池", FONTS["h2"], INK)
    centered(draw, (1045, 365), str(current), FONTS["number"], GREEN)
    centered(draw, (1045, 430), "TOP300", FONTS["h1"], GREEN)
    draw.text((948, 477), "只判断文本相关性", font=FONTS["small"], fill=MUTED)


def draw_scene_4(draw, t: float) -> None:
    p = scene_progress(t, 23, 32)
    header(draw, 3, "第三步：TOP300 收束为候选 TOP30", "岗位名称去重 + 岗位族配额控制")
    draw.text((55, 160), "TOP300", font=FONTS["h1"], fill=BLUE)
    duplicate_jobs = [
        "Java后端工程师",
        "JAVA 后端开发工程师",
        "Java后端工程师（高级）",
        "数据开发工程师",
        "软件测试工程师",
        "云运维工程师",
    ]
    for i, name in enumerate(duplicate_jobs):
        y = 225 + i * 62
        rounded(draw, (55, y, 340, y + 44), WHITE, LINE, 12, 2)
        draw.text((72, y + 10), name, font=FONTS["small"], fill=INK)

    arrow(draw, (370, 370), (485, 370), BLUE, 7)
    rounded(draw, (500, 165, 800, 590), WHITE, LINE, 28, 2)
    centered(draw, (650, 205), "去重与分组", FONTS["h2"], INK)
    families = [
        ("后端", BLUE),
        ("数据", CYAN),
        ("测试", GREEN),
        ("运维", ORANGE),
        ("算法", RED),
        ("前端", (123, 91, 210)),
    ]
    for i, (name, color) in enumerate(families):
        row, col = divmod(i, 2)
        x = 545 + col * 130
        y = 270 + row * 88
        rounded(draw, (x, y, x + 105, y + 56), (*color, 25), color, 16, 2)
        centered(draw, (x + 52, y + 28), name, FONTS["body"], color)
    fit_text(
        draw,
        "同名变体合并；目标岗位族最多24个，其他岗位族最多8个",
        (535, 520, 765, 580),
        FONTS["small"],
        MUTED,
        align="center",
    )
    arrow(draw, (820, 370), (930, 370), BLUE, 7)
    rounded(draw, (945, 220, 1190, 510), PALE_GREEN, GREEN, 28, 3)
    centered(draw, (1067, 285), "候选池", FONTS["h2"], INK)
    displayed = round(300 - 270 * ease(clamp((p - 0.25) / 0.6)))
    centered(draw, (1067, 380), str(displayed), FONTS["number"], GREEN)
    centered(draw, (1067, 450), "TOP30", FONTS["h1"], GREEN)


LOCAL_TOP10 = [
    ("软件开发工程师", 72, BLUE),
    ("云计算工程师", 66, CYAN),
    ("JAVA后端开发工程师", 57, GREEN),
    ("大数据后端开发", 56, ORANGE),
    ("数据库开发工程师", 52, BLUE),
    ("DevOps工程师", 52, CYAN),
    ("运维工程师", 47, GREEN),
    ("Java全栈开发工程师", 47, ORANGE),
    ("Web3运维/DevOps", 46, BLUE),
    ("IoT后端工程师", 45, CYAN),
]


def draw_scene_5(draw, t: float) -> None:
    p = scene_progress(t, 32, 41)
    header(draw, 3, "第四步：六项规则评分形成本地 TOP10", "高分优先，同时控制岗位名称重复和岗位族集中")
    rounded(draw, (45, 155, 600, 630), WHITE, LINE, 26, 2)
    draw.text((75, 185), "本地匹配分（100分）", font=FONTS["h2"], fill=INK)
    parts = [
        ("技能覆盖", 55, BLUE),
        ("项目证据", 15, CYAN),
        ("能力画像", 15, GREEN),
        ("目标方向", 10, ORANGE),
        ("学历适配", 3, RED),
        ("证书适配", 2, (123, 91, 210)),
    ]
    for i, (name, weight, color) in enumerate(parts):
        y = 250 + i * 54
        draw.text((80, y), name, font=FONTS["small"], fill=INK)
        draw.rounded_rectangle((215, y + 3, 500, y + 27), radius=12, fill=(227, 235, 247))
        progress = ease(clamp((p - i * 0.05) / 0.45))
        fill_w = int(285 * weight / 55 * progress)
        draw.rounded_rectangle((215, y + 3, 215 + fill_w, y + 27), radius=12, fill=color)
        draw.text((520, y), f"×{weight}", font=FONTS["small"], fill=color)
    rounded(draw, (75, 575, 570, 610), PALE_ORANGE, ORANGE, 12, 2)
    centered(draw, (322, 592), "质量门槛 = max(40，最高分−25)", FONTS["small"], ORANGE)

    rounded(draw, (630, 155, 1235, 630), WHITE, LINE, 26, 2)
    draw.text((665, 185), "本地 TOP10", font=FONTS["h2"], fill=INK)
    for i, (name, score, color) in enumerate(LOCAL_TOP10):
        row, col = divmod(i, 2)
        x = 665 + col * 270
        y = 245 + row * 68
        appear = ease(clamp((p - i * 0.035) / 0.35))
        rounded(
            draw,
            (x, y, x + 250, y + 50),
            (*color, int(20 * appear)),
            (*color, int(210 * appear)),
            13,
            2,
        )
        draw.text((x + 12, y + 8), f"{i+1}", font=FONTS["small"], fill=(*color, int(255 * appear)))
        name_font = FONTS["tiny"] if len(name) > 10 else FONTS["small"]
        draw.text((x + 43, y + 8), name, font=name_font, fill=(*INK, int(255 * appear)))
        draw.text((x + 205, y + 8), str(score), font=FONTS["small"], fill=(*color, int(255 * appear)))
    centered(draw, (930, 598), "只将这10个岗位交给AI", FONTS["body"], GREEN)


FINAL_TOP5 = [
    ("软件开发工程师", 83, "综合技能与项目高度吻合"),
    ("JAVA后端开发工程师", 73, "Java技术栈方向明确"),
    ("Java全栈开发工程师", 63, "兼具后端与前端基础"),
    ("云计算工程师", 58, "具备Linux与容器化经验"),
    ("运维工程师", 47, "部署与基础设施能力可迁移"),
]


def draw_scene_6(draw, t: float) -> None:
    p = scene_progress(t, 41, 48)
    header(draw, 4, "第五步：AI 对本地 TOP10 一次性精排", "理解岗位要求与简历证据，重新校准顺序和分数")
    rounded(draw, (45, 150, 355, 635), WHITE, LINE, 24, 2)
    centered(draw, (200, 190), "本地候选", FONTS["h2"], INK)
    for i, (name, score, color) in enumerate(LOCAL_TOP10):
        y = 235 + i * 36
        rounded(draw, (75, y, 325, y + 28), (*color, 20), (*color, 180), 8, 1)
        draw.text((85, y + 5), f"{i+1}. {name}", font=FONTS["tiny"], fill=INK)

    pulse = 0.8 + 0.2 * math.sin(t * 5) ** 2
    for radius, alpha in [(145, 15), (115, 30), (88, 60)]:
        draw.ellipse(
            (640 - radius, 380 - radius, 640 + radius, 380 + radius),
            fill=(*BLUE, int(alpha * pulse)),
            outline=(*BLUE, int(120 * pulse)),
            width=3,
        )
    draw.ellipse((565, 305, 715, 455), fill=BLUE)
    centered(draw, (640, 355), "AI", FONTS["hero"], WHITE)
    centered(draw, (640, 410), "双向精排", FONTS["body"], WHITE)
    arrow(draw, (375, 380), (500, 380), BLUE, 7)
    arrow(draw, (780, 380), (905, 380), GREEN, 7)
    draw.text((440, 330), "10个岗位", font=FONTS["small"], fill=BLUE)
    draw.text((805, 330), "重新排序", font=FONTS["small"], fill=GREEN)

    rounded(draw, (925, 150, 1235, 635), WHITE, LINE, 24, 2)
    centered(draw, (1080, 190), "精排结果", FONTS["h2"], INK)
    for i, (name, score, _) in enumerate(FINAL_TOP5):
        appear = ease(clamp((p - 0.18 - i * 0.08) / 0.28))
        y = 245 + i * 72
        x = 950 + (1 - appear) * 120
        color = [ORANGE, BLUE, CYAN, GREEN, (123, 91, 210)][i]
        rounded(draw, (int(x), y, 1208, y + 54), (*color, 24), (*color, int(230 * appear)), 14, 2)
        draw.text((x + 12, y + 13), f"{i+1}", font=FONTS["body"], fill=(*color, int(255 * appear)))
        name_font = FONTS["tiny"] if len(name) > 10 else FONTS["small"]
        draw.text((x + 48, y + 12), name, font=name_font, fill=(*INK, int(255 * appear)))
        draw.text((1160, y + 12), str(score), font=FONTS["small"], fill=(*color, int(255 * appear)))


def draw_scene_7(draw, t: float) -> None:
    p = scene_progress(t, 48, 55)
    header(draw, 5, "最终推荐 TOP5", "来自真实679条岗位库与当前项目匹配流程")
    rounded(draw, (80, 145, 1200, 620), WHITE, LINE, 30, 2)
    for i, (name, score, reason) in enumerate(FINAL_TOP5):
        appear = ease(clamp((p - i * 0.07) / 0.35))
        y = 195 + i * 78
        x = 120 + (1 - appear) * 180
        color = [ORANGE, BLUE, CYAN, GREEN, (123, 91, 210)][i]
        draw.ellipse((x, y, x + 54, y + 54), fill=(*color, int(255 * appear)))
        centered(draw, (x + 27, y + 27), str(i + 1), FONTS["body"], (*WHITE, int(255 * appear)))
        draw.text((x + 78, y + 3), name, font=FONTS["body"], fill=(*INK, int(255 * appear)))
        draw.text((x + 78, y + 34), reason, font=FONTS["small"], fill=(*MUTED, int(255 * appear)))
        bar_x = 700
        bar_y = y + 15
        draw.rounded_rectangle((bar_x, bar_y, 1070, bar_y + 22), radius=11, fill=(226, 234, 246))
        draw.rounded_rectangle(
            (bar_x, bar_y, bar_x + int(370 * score / 100 * appear), bar_y + 22),
            radius=11,
            fill=color,
        )
        draw.text((1090, y + 10), f"{score}分", font=FONTS["body"], fill=(*color, int(255 * appear)))
    centered(
        draw,
        (640, 650),
        "简历输入 → 679 → TOP300 → TOP30 → 本地TOP10 → AI精排 → TOP5",
        FONTS["body"],
        BLUE,
    )


def render_frame(t: float) -> Image.Image:
    image = background()
    draw = ImageDraw.Draw(image, "RGBA")
    if t < 7:
        draw_scene_1(draw, t)
    elif t < 14:
        draw_scene_2(draw, t)
    elif t < 23:
        draw_scene_3(draw, t)
    elif t < 32:
        draw_scene_4(draw, t)
    elif t < 41:
        draw_scene_5(draw, t)
    elif t < 48:
        draw_scene_6(draw, t)
    else:
        draw_scene_7(draw, t)
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUTPUT),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame_index in range(FRAME_COUNT):
        frame = render_frame(frame_index / FPS)
        process.stdin.write(frame.tobytes())
        if frame_index % (FPS * 5) == 0:
            print(f"rendered {frame_index / FPS:.0f}s / {DURATION:.0f}s", flush=True)
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)
    print(OUTPUT)


if __name__ == "__main__":
    main()
