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

MICROSERVICE_RULE = {
    "id": "rule_016",
    "category": "microservice_architecture",
    "title": "微服务项目需要说明架构拆分和个人负责模块",
    "problem_patterns": ["微服务", "Spring Cloud", "Spring Cloud Alibaba", "Nacos", "Gateway", "OpenFeign"],
    "suggestion": "微服务项目应说明系统拆分和个人负责模块。",
}

GATEWAY_RULE = {
    "id": "rule_017",
    "category": "microservice_architecture",
    "title": "Nacos / Gateway 需要说明用途",
    "problem_patterns": ["Nacos", "Gateway", "注册配置中心", "统一网关", "路由", "跨域"],
    "suggestion": "应说明 Nacos 和 Gateway 的项目作用。",
}

SECURITY_RULE = {
    "id": "rule_018",
    "category": "security",
    "title": "Spring Security / JWT 需要说明安全场景",
    "problem_patterns": ["Spring Security", "JWT", "认证", "授权", "接口安全"],
    "suggestion": "安全相关技术应说明认证授权和接口安全场景。",
}

XXL_JOB_RULE = {
    "id": "rule_019",
    "category": "scheduled_task",
    "title": "XXL-JOB 需要说明解决的问题",
    "problem_patterns": ["XXL-JOB", "定时任务", "分片广播", "订单状态", "自动处理"],
    "suggestion": "XXL-JOB 应说明解决的定时任务或状态一致性问题。",
}

DOCKER_RULE = {
    "id": "rule_020",
    "category": "deployment",
    "title": "Docker Compose 需要说明部署价值",
    "problem_patterns": ["Docker", "Docker Compose", "部署", "运维", "测试环境"],
    "suggestion": "部署相关内容应说明部署价值。",
}

ROLE_RULE = {
    "id": "rule_021",
    "category": "project_experience",
    "title": "个人角色需要更突出",
    "problem_patterns": ["个人角色", "后端负责人", "核心后端开发", "独立完成", "主要负责"],
    "suggestion": "项目经历中应突出个人角色和负责范围。",
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


def test_highlights_microservice_architecture_terms():
    html = build_highlighted_optimized_html(
        "",
        "基于 Spring Cloud Alibaba 搭建微服务生态，并通过 OpenFeign 实现服务间调用。",
        [MICROSERVICE_RULE],
    )

    assert 'title="微服务项目需要说明架构拆分和个人负责模块"' in html
    assert "Spring Cloud Alibaba" in html
    assert "expert-highlight" in html


def test_highlights_nacos_and_gateway_usage():
    html = build_highlighted_optimized_html(
        "",
        "使用 Nacos 作为注册配置中心，Gateway 实现统一网关、路由转发与跨域处理。",
        [GATEWAY_RULE],
    )

    assert 'title="Nacos / Gateway 需要说明用途"' in html
    assert ">Nacos 作为注册配置中心<" in html
    assert "Gateway 实现统一网关" in html


def test_highlights_jwt_security_scene():
    html = build_highlighted_optimized_html(
        "",
        "基于 Spring Security + JWT 完成认证授权和接口安全控制。",
        [SECURITY_RULE],
    )

    assert 'title="Spring Security / JWT 需要说明安全场景"' in html
    assert "Spring Security + JWT" in html
    assert "认证授权" in html


def test_highlights_xxl_job_problem_solution():
    html = build_highlighted_optimized_html(
        "",
        "使用 XXL-JOB 分片广播解决订单状态不一致问题，实现超时订单自动处理。",
        [XXL_JOB_RULE],
    )

    assert 'title="XXL-JOB 需要说明解决的问题"' in html
    assert "XXL-JOB 分片广播" in html
    assert "订单状态" in html


def test_highlights_docker_compose_deployment_value():
    html = build_highlighted_optimized_html(
        "",
        "使用 Docker Compose 简化测试环境搭建，提高部署效率。",
        [DOCKER_RULE],
    )

    assert 'title="Docker Compose 需要说明部署价值"' in html
    assert ">Docker Compose 简化测试环境搭建<" in html
    assert ">提高部署效率<" in html


def test_highlights_role_and_non_percentage_effect_phrase():
    html = build_highlighted_optimized_html(
        "",
        "本人作为后端核心开发，主要负责订单服务，降低数据库重复访问、提高响应速度。",
        [ROLE_RULE, QUANT_RULE],
    )

    assert 'title="个人角色需要更突出"' in html
    assert "后端核心开发" in html
    assert 'title="尽量补充量化结果"' in html
    assert "降低数据库重复访问" in html
