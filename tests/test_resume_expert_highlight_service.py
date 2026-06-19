from app.services.resume_optimizer_service import build_highlighted_optimized_html


QUANT_RULE = {
    "id": "rule_006",
    "category": "technical_highlight",
    "title": "尽量补充量化结果",
    "problem_patterns": ["提升", "优化", "效率", "性能", "响应时间", "减少"],
    "suggestion": "能量化的地方尽量量化。",
}

REDIS_RULE = {
    "id": "rule_007",
    "category": "technical_highlight",
    "title": "Redis 缓存需要说明用途",
    "problem_patterns": ["Redis", "缓存", "查询效率", "高频查询"],
    "suggestion": "写 Redis 时应说明缓存对象和效果。",
}

ENGLISH_RULE = {
    "id": "rule_013",
    "category": "certificates",
    "title": "英语四六级需要写分数",
    "problem_patterns": ["英语四级", "英语六级", "CET-4", "CET-6", "四六级"],
    "suggestion": "英语四六级建议写明分数。",
}

PROJECT_TIME_RULE = {
    "id": "rule_002",
    "category": "project_experience",
    "title": "项目经历需要补充项目时间",
    "problem_patterns": ["项目经历", "项目时间", "起止时间", "开发周期"],
    "suggestion": "项目经历应补充起止时间。",
}


def test_highlights_quantified_percentage_result():
    html = build_highlighted_optimized_html(
        "",
        "通过缓存优化使响应时间降低40%。",
        [QUANT_RULE],
    )

    assert 'class="expert-highlight"' in html
    assert 'title="尽量补充量化结果"' in html
    assert ">响应时间降低40%<" in html


def test_highlights_redis_usage_phrase():
    html = build_highlighted_optimized_html(
        "",
        "使用 Redis 缓存高频查询结果，减少数据库重复访问。",
        [REDIS_RULE],
    )

    assert 'class="expert-highlight"' in html
    assert 'title="Redis 缓存需要说明用途"' in html
    assert ">使用 Redis 缓存高频查询结果<" in html


def test_highlights_english_score():
    html = build_highlighted_optimized_html(
        "",
        "证书：CET-6 455分，具备英文文档阅读能力。",
        [ENGLISH_RULE],
    )

    assert 'class="expert-highlight"' in html
    assert 'title="英语四六级需要写分数"' in html
    assert ">CET-6 455分<" in html


def test_without_expert_rules_returns_escaped_text_without_highlight():
    html = build_highlighted_optimized_html(
        "",
        "后端核心成员 <b>优化</b>",
        [],
    )

    assert 'class="expert-highlight"' not in html
    assert html == "后端核心成员 &lt;b&gt;优化&lt;/b&gt;"


def test_escapes_html_special_characters_to_avoid_xss():
    html = build_highlighted_optimized_html(
        "",
        "项目 <script>alert(1)</script>\n响应时间降低40%",
        [QUANT_RULE],
    )

    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert 'class="expert-highlight"' in html


def test_existing_project_time_is_not_automatically_highlighted():
    html = build_highlighted_optimized_html(
        "Linux 文件服务器｜2024.03-2024.06",
        "Linux 文件服务器｜2024.03-2024.06\n主要负责后端接口开发。",
        [PROJECT_TIME_RULE],
    )

    assert 'class="expert-highlight"' not in html
    assert "2024.03-2024.06" in html


def test_new_project_time_can_be_highlighted_when_absent_from_original():
    html = build_highlighted_optimized_html(
        "Linux 文件服务器",
        "Linux 文件服务器｜2024.03-2024.06\n主要负责后端接口开发。",
        [PROJECT_TIME_RULE],
    )

    assert 'class="expert-highlight"' in html
    assert 'title="项目经历需要补充项目时间"' in html
    assert ">2024.03-2024.06<" in html
