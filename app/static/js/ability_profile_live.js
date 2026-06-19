(function () {
    "use strict";

    const form = document.getElementById("liveProfileForm");
    if (!form) {
        return;
    }

    const fileInput = document.getElementById("liveResumeFile");
    const dropZone = document.getElementById("liveDropZone");
    const dropText = document.getElementById("liveDropText");
    const generateButton = document.getElementById("liveGenerateButton");
    const liveState = document.getElementById("liveState");
    const exportForm = document.getElementById("profileExportForm");
    const exportButton = document.getElementById("profileExportButton");
    const jobMatchAction = document.getElementById("jobMatchAction");
    const employmentAction = document.getElementById("employmentAction");

    let selectedFile = null;
    let renderedToolCalls = 0;
    let renderedHandoffs = 0;
    let cacheReplay = false;

    function setState(text, state) {
        liveState.textContent = text;
        liveState.className = `live-state ${state || ""}`.trim();
    }

    function activate(element) {
        const panel = element && element.closest(".profile-panel");
        if (!panel) {
            return;
        }
        panel.classList.add("live-panel-active");
        window.setTimeout(() => panel.classList.remove("live-panel-active"), 1100);
    }

    function typingDelay(text) {
        if (!text) {
            return 0;
        }
        const targetDuration = cacheReplay ? 140 : 520;
        const maximumDelay = cacheReplay ? 6 : 14;
        return Math.max(1, Math.min(maximumDelay, Math.floor(targetDuration / text.length)));
    }

    async function typeText(element, value) {
        if (!element) {
            return;
        }
        const text = String(value == null || value === "" ? "无" : value);
        element.textContent = "";
        element.classList.remove("live-placeholder", "live-placeholder-block");
        element.classList.add("live-typing");
        activate(element);

        const delay = typingDelay(text);
        for (let index = 0; index < text.length; index += 1) {
            element.textContent += text[index];
            if (delay) {
                await new Promise((resolve) => window.setTimeout(resolve, delay));
            }
        }
        element.classList.remove("live-typing");
    }

    function createElement(tag, className, text) {
        const element = document.createElement(tag);
        if (className) {
            element.className = className;
        }
        if (text != null) {
            element.textContent = text;
        }
        return element;
    }

    async function renderTextList(containerId, items, listClass) {
        const container = document.getElementById(containerId);
        if (!container || !Array.isArray(items)) {
            return;
        }
        container.replaceChildren();
        const list = createElement("ul", listClass || "analysis-list");
        container.appendChild(list);
        for (const item of items) {
            const li = createElement("li");
            list.appendChild(li);
            await typeText(li, item);
        }
    }

    async function renderStudent(student) {
        const fields = [
            ["studentName", student.name],
            ["studentMajor", student.major],
            ["studentGrade", student.grade],
            ["studentTargetJob", student.target_job],
        ];
        for (const [id, value] of fields) {
            await typeText(document.getElementById(id), value);
        }
    }

    async function renderScores(scores) {
        if (!scores) {
            return;
        }
        const dimensions = ["professional", "practice", "tools", "career"];
        for (const key of dimensions) {
            const value = Number(scores[key]) || 0;
            await typeText(document.getElementById(`${key}Score`), `${value} 分`);
            const progress = document.getElementById(`${key}Progress`);
            if (progress) {
                progress.style.width = `${value}%`;
            }
        }
        if (typeof window.updateAbilityRadar === "function") {
            window.updateAbilityRadar(scores);
        }
    }

    async function renderTags(tags) {
        const container = document.getElementById("profileTags");
        if (!container || !Array.isArray(tags)) {
            return;
        }
        container.replaceChildren();
        container.style.display = tags.length ? "flex" : "none";
        for (const tag of tags) {
            const item = createElement("span", "profile-tag");
            container.appendChild(item);
            await typeText(item, tag);
        }
    }

    async function appendToolCalls(toolCalls) {
        if (!Array.isArray(toolCalls) || toolCalls.length <= renderedToolCalls) {
            return;
        }
        const container = document.getElementById("toolCallsContainer");
        let list = container.querySelector(".tool-call-list");
        if (!list) {
            container.replaceChildren();
            list = createElement("ul", "tool-call-list");
            container.appendChild(list);
        }

        for (const call of toolCalls.slice(renderedToolCalls)) {
            const item = createElement("li", "tool-call-item");
            const head = createElement("div", "tool-call-head");
            const name = createElement("strong");
            const agent = createElement("span");
            head.append(name, agent);
            item.appendChild(head);
            list.appendChild(item);
            await typeText(name, call.tool_name);
            await typeText(agent, call.called_by);

            for (const [label, value] of [
                ["用途：", call.purpose],
                ["输入：", call.input_summary],
                ["输出：", call.output_summary],
            ]) {
                const paragraph = createElement("p");
                paragraph.appendChild(createElement("strong", "", label));
                const content = createElement("span");
                paragraph.appendChild(content);
                item.appendChild(paragraph);
                await typeText(content, value);
            }
        }
        renderedToolCalls = toolCalls.length;
        document.getElementById("toolCallCount").textContent = String(renderedToolCalls);
    }

    async function appendHandoffs(handoffs) {
        if (!Array.isArray(handoffs) || handoffs.length <= renderedHandoffs) {
            return;
        }
        const container = document.getElementById("handoffContainer");
        let list = container.querySelector(".handoff-list");
        if (!list) {
            container.replaceChildren();
            list = createElement("ul", "handoff-list");
            container.appendChild(list);
        }

        for (const handoff of handoffs.slice(renderedHandoffs)) {
            const item = createElement("li", "handoff-item");
            const head = createElement("div", "handoff-head");
            const route = createElement("strong");
            const artifact = createElement("span");
            const message = createElement("p");
            head.append(route, artifact);
            item.append(head, message);
            list.appendChild(item);
            await typeText(route, `${handoff.sender} → ${handoff.receiver}`);
            await typeText(artifact, handoff.artifact);
            await typeText(message, handoff.message);
        }
        renderedHandoffs = handoffs.length;
        document.getElementById("handoffCount").textContent = String(renderedHandoffs);
    }

    async function renderEvidenceCards(cards) {
        if (!Array.isArray(cards)) {
            return;
        }
        const container = document.getElementById("evidenceCardsContainer");
        container.replaceChildren();
        const grid = createElement("div", "evidence-grid");
        container.appendChild(grid);

        for (const card of cards) {
            const article = createElement("article", "evidence-item");
            const heading = createElement("strong");
            const name = createElement("span");
            const confidence = createElement("span", "confidence");
            heading.append(name, confidence);
            const interpretation = createElement("p");
            const evidenceList = createElement("ul", "mini-list");
            article.append(heading, interpretation, evidenceList);
            grid.appendChild(article);
            await typeText(name, card.name);
            await typeText(confidence, `可信度 ${card.confidence || "待复核"}`);
            await typeText(interpretation, card.interpretation);
            for (const evidence of (card.evidence || []).slice(0, 4)) {
                const li = createElement("li");
                evidenceList.appendChild(li);
                await typeText(li, evidence);
            }
        }
    }

    async function renderDimensions(items) {
        if (!Array.isArray(items)) {
            return;
        }
        const container = document.getElementById("dimensionInsightsContainer");
        container.replaceChildren();
        const grid = createElement("div", "dimension-grid");
        container.appendChild(grid);

        for (const item of items) {
            const article = createElement("article", "dimension-item");
            const head = createElement("div", "dimension-head");
            const name = createElement("strong");
            const score = createElement("span", "score-badge");
            const level = createElement("p", "level-line");
            const conclusion = createElement("p");
            const evidenceList = createElement("ul", "mini-list");
            const actionLine = createElement("p");
            actionLine.appendChild(createElement("strong", "", "下一步："));
            const action = createElement("span");
            actionLine.appendChild(action);
            head.append(name, score);
            article.append(head, level, conclusion, evidenceList, actionLine);
            grid.appendChild(article);
            await typeText(name, item.name);
            await typeText(score, `${item.score} 分`);
            await typeText(level, item.level);
            await typeText(conclusion, item.conclusion);
            for (const evidence of (item.evidence || []).slice(0, 3)) {
                const li = createElement("li");
                evidenceList.appendChild(li);
                await typeText(li, evidence);
            }
            await typeText(action, item.next_action);
        }
    }

    async function renderFocus(items) {
        if (!Array.isArray(items)) {
            return;
        }
        const container = document.getElementById("developmentFocusContainer");
        container.replaceChildren();
        const grid = createElement("div", "focus-grid");
        container.appendChild(grid);
        for (const item of items) {
            const article = createElement("article", "focus-item");
            const heading = createElement("strong");
            const name = createElement("span");
            const priority = createElement("span", "priority");
            const reason = createElement("p");
            const actionLine = createElement("p");
            actionLine.appendChild(createElement("strong", "", "动作："));
            const action = createElement("span");
            actionLine.appendChild(action);
            heading.append(name, priority);
            article.append(heading, reason, actionLine);
            grid.appendChild(article);
            await typeText(name, item.name);
            await typeText(priority, `${item.priority || "中"}优先级`);
            await typeText(reason, item.reason);
            await typeText(action, item.action);
        }
    }

    async function renderFindings(findings) {
        const container = document.getElementById("reviewFindingsContainer");
        if (!container || !Array.isArray(findings) || !findings.length) {
            return;
        }
        container.replaceChildren();
        container.appendChild(createElement("h3", "", "审计发现"));
        const list = createElement("ul", "finding-list");
        container.appendChild(list);
        for (const finding of findings) {
            const item = createElement("li", `finding-item ${finding.severity || "low"}`);
            const title = createElement("strong");
            const text = createElement("p");
            item.append(title, text);
            list.appendChild(item);
            await typeText(title, finding.dimension);
            await typeText(text, finding.finding);
        }
    }

    async function handleAgentStep(event) {
        const data = event.data || {};
        setState(event.title || "智能体生成中", "running");

        await appendToolCalls(data.tool_calls);
        await appendHandoffs(data.collaboration_log);
        if (Array.isArray(data.llm_agents)) {
            document.getElementById("llmAgentCount").textContent = String(data.llm_agents.length);
        }

        if (event.node === "score_ability") {
            await renderScores(data.ability_scores || event.ability_scores);
        } else if (event.node === "analyze_profile_evidence") {
            await renderTags(data.profile_tags || []);
            await renderEvidenceCards(data.evidence_cards || []);
            await renderTextList("riskFlagsContainer", data.risk_flags || [], "risk-list");
        } else if (event.node === "diagnose_ability") {
            await typeText(document.getElementById("profileSummary"), data.summary || event.summary);
            await renderTextList("advantagesContainer", data.advantages || [], "analysis-list");
            await renderTextList("weaknessesContainer", data.weaknesses || [], "analysis-list");
            await renderDimensions(data.dimension_insights || []);
            await renderFocus(data.development_focus || []);
        } else if (event.node === "review_profile") {
            await renderTextList("qualityReviewContainer", data.quality_review || [], "review-list");
            await renderFindings(data.review_findings || []);
        }
    }

    async function handleEvent(event) {
        if (event.type === "accepted") {
            cacheReplay = Boolean(event.cache_hit);
            if (cacheReplay && selectedFile) {
                dropText.textContent = `缓存命中：${selectedFile.name}`;
            }
            setState(
                cacheReplay ? "已命中缓存，正在快速加载" : "正在解析简历",
                "running"
            );
            return;
        }
        if (event.type === "profile") {
            setState("正在生成基础信息", "running");
            await renderStudent(event.student || {});
            return;
        }
        if (event.type === "agent_step") {
            await handleAgentStep(event);
            return;
        }
        if (event.type === "complete") {
            const metrics = event.metrics || {};
            document.getElementById("agentRosterCount").textContent = String(metrics.agent_roster || 5);
            document.getElementById("llmAgentCount").textContent = String(metrics.llm_agents || 0);
            document.getElementById("toolCallCount").textContent = String(metrics.tool_calls || renderedToolCalls);
            document.getElementById("handoffCount").textContent = String(metrics.collaboration_log || renderedHandoffs);
            setState(event.cache_hit ? "缓存画像加载完成" : "生成完成", "complete");
            exportForm.action = `${event.redirect_url}/export-json`;
            exportButton.disabled = false;
            jobMatchAction.classList.remove("profile-action-disabled");
            employmentAction.classList.remove("profile-action-disabled");
            window.history.replaceState({}, "", event.redirect_url);
            return;
        }
        if (event.type === "error") {
            setState(event.text || "生成失败", "error");
            generateButton.disabled = false;
            fileInput.disabled = false;
            generateButton.textContent = "重新生成";
        }
    }

    async function consumeNdjson(response) {
        if (!response.body) {
            throw new Error("浏览器不支持流式响应");
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let renderChain = Promise.resolve();

        while (true) {
            const chunk = await reader.read();
            buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            for (const line of lines.filter(Boolean)) {
                const event = JSON.parse(line);
                renderChain = renderChain.then(() => handleEvent(event));
            }
            if (chunk.done) {
                break;
            }
        }
        if (buffer.trim()) {
            const event = JSON.parse(buffer);
            renderChain = renderChain.then(() => handleEvent(event));
        }
        await renderChain;
    }

    function rememberFile(file) {
        selectedFile = file || null;
        generateButton.disabled = !selectedFile;
        dropZone.classList.toggle("has-file", Boolean(selectedFile));
        dropText.textContent = selectedFile
            ? `已选择：${selectedFile.name}`
            : "拖入或点击选择简历（PDF、DOCX、TXT、MD、CSV）";
        setState(selectedFile ? "等待生成" : "等待上传", "");
    }

    fileInput.addEventListener("change", () => rememberFile(fileInput.files[0]));
    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.add("dragging");
        });
    });
    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.remove("dragging");
        });
    });
    dropZone.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files[0];
        if (file) {
            rememberFile(file);
        }
    });

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!selectedFile) {
            return;
        }
        generateButton.disabled = true;
        fileInput.disabled = true;
        generateButton.textContent = "生成中…";
        setState("智能体启动中", "running");

        const formData = new FormData();
        formData.append("resume_file", selectedFile);
        try {
            const response = await fetch("/ability/profile/generate", {
                method: "POST",
                body: formData,
            });
            await consumeNdjson(response);
        } catch (error) {
            setState("连接中断，请重试", "error");
            generateButton.disabled = false;
            fileInput.disabled = false;
            generateButton.textContent = "重新生成";
        }
    });
})();
