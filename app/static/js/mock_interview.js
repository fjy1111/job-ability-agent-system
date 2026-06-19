(() => {
    "use strict";

    const examDataNode = document.getElementById("writtenExamData");
    if (!examDataNode) {
        return;
    }

    const examSession = JSON.parse(examDataNode.textContent);
    const writtenPhase = document.getElementById("writtenPhase");
    const writtenResultPhase = document.getElementById("writtenResultPhase");
    const videoPhase = document.getElementById("videoPhase");
    const writtenForm = document.getElementById("writtenExamForm");
    const writtenAnsweredCount = document.getElementById("writtenAnsweredCount");
    const submitWrittenButton = document.getElementById("submitWrittenButton");
    const writtenReviewList = document.getElementById("writtenReviewList");
    const startVideoButton = document.getElementById("startVideoInterviewButton");
    const candidateVideo = document.getElementById("candidateVideo");
    const cameraPlaceholder = document.getElementById("cameraPlaceholder");
    const cameraBadge = document.getElementById("cameraBadge");
    const enableCameraButton = document.getElementById("enableCameraButton");
    const expressionStatus = document.getElementById("expressionStatus");
    const expressionLabel = document.getElementById("expressionLabel");
    const tensionValue = document.getElementById("tensionValue");
    const tensionBar = document.getElementById("tensionBar");
    const calmReminder = document.getElementById("calmReminder");
    const conversationList = document.getElementById("conversationList");
    const adviceContent = document.getElementById("adviceContent");
    const answerForm = document.getElementById("answerForm");
    const answerText = document.getElementById("answerText");
    const submitAnswerButton = document.getElementById("submitAnswerButton");
    const roundLabel = document.getElementById("roundLabel");
    const roleLabel = document.getElementById("roleLabel");
    const roundProgress = document.getElementById("roundProgress");

    let interviewSession = null;
    let interviewHistory = [];
    let cameraStream = null;
    let expressionTimer = null;
    let expressionBusy = false;
    let smoothedTension = 0;
    let tenseStreak = 0;
    let lastReminderAt = 0;
    let faceMissCount = 0;
    let faceModelPromise = null;

    const FACE_API_SCRIPT = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.js";
    const FACE_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model";
    const expressionNames = {
        neutral: "自然专注",
        happy: "轻松微笑",
        sad: "略显低落",
        angry: "眉眼紧绷",
        fearful: "紧张趋势",
        disgusted: "表情紧绷",
        surprised: "略显惊讶"
    };

    function escapeHtml(value) {
        return String(value || "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function setStep(step) {
        const order = ["written", "video", "review"];
        const activeIndex = order.indexOf(step);
        document.querySelectorAll("#interviewStepper [data-step]").forEach((item) => {
            const index = order.indexOf(item.dataset.step);
            item.classList.toggle("active", index === activeIndex);
            item.classList.toggle("completed", index < activeIndex);
        });
    }

    function collectWrittenAnswers() {
        const answers = {};
        examSession.questions.forEach((question) => {
            const selected = writtenForm.querySelector(`input[name="${question.id}"]:checked`);
            if (selected) {
                answers[question.id] = Number(selected.value);
            }
        });
        return answers;
    }

    function updateAnsweredCount() {
        const count = Object.keys(collectWrittenAnswers()).length;
        writtenAnsweredCount.textContent = `已作答 ${count} / ${examSession.total_questions}`;
    }

    function renderWrittenResult(result) {
        document.getElementById("writtenScore").textContent = result.score;
        document.getElementById("writtenScoreRing").style.setProperty("--score", `${result.score * 3.6}deg`);
        document.getElementById("writtenResultTitle").textContent = result.score >= 60 ? "笔试通过，准备进入面试" : "笔试完成，继续用面试展现自己";
        document.getElementById("writtenResultSummary").textContent = `答对 ${result.correct_count} / ${result.total_questions} 题。下面可以查看逐题解析，准备好后开启视频面试。`;
        writtenReviewList.innerHTML = "";

        result.details.forEach((detail, index) => {
            const item = document.createElement("article");
            item.className = `written-review-item ${detail.is_correct ? "correct" : "incorrect"}`;
            item.innerHTML = `
                <div class="written-review-number">${String(index + 1).padStart(2, "0")}</div>
                <div>
                    <span>${escapeHtml(detail.category)} · ${detail.is_correct ? "回答正确" : "需要复习"}</span>
                    <h3>${escapeHtml(detail.question)}</h3>
                    <p>你的答案：${escapeHtml(detail.selected_answer)}</p>
                    ${detail.is_correct ? "" : `<p>正确答案：${escapeHtml(detail.correct_answer)}</p>`}
                    <small>${escapeHtml(detail.explanation)}</small>
                </div>
            `;
            writtenReviewList.appendChild(item);
        });
    }

    writtenForm.addEventListener("change", updateAnsweredCount);
    writtenForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const answers = collectWrittenAnswers();
        if (Object.keys(answers).length !== examSession.total_questions) {
            const firstUnanswered = examSession.questions.find((question) => !(question.id in answers));
            const card = writtenForm.querySelector(`[data-question-id="${firstUnanswered.id}"]`);
            card.classList.add("needs-answer");
            card.scrollIntoView({ behavior: "smooth", block: "center" });
            window.setTimeout(() => card.classList.remove("needs-answer"), 1800);
            return;
        }

        submitWrittenButton.disabled = true;
        submitWrittenButton.textContent = "正在批改并生成面试题...";
        try {
            const response = await fetch("/interview/written/submit", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ exam_session: examSession, answers })
            });
            const data = await response.json();
            if (!data.ok) {
                throw new Error((data.errors || ["笔试提交失败，请稍后重试。"]).join(" "));
            }

            interviewSession = data.session;
            interviewHistory = [];
            renderWrittenResult(data.written_result);
            writtenPhase.hidden = true;
            writtenResultPhase.hidden = false;
            startVideoButton.dataset.openingMessage = data.opening_message;
            startVideoButton.dataset.firstQuestion = data.question;
            setStep("video");
            window.scrollTo({ top: 0, behavior: "smooth" });
        } catch (error) {
            window.alert(error.message || "笔试提交失败，请检查服务是否正在运行。");
            submitWrittenButton.disabled = false;
            submitWrittenButton.textContent = "提交笔试";
        }
    });

    function appendMessage(role, text, label) {
        const message = document.createElement("article");
        message.className = `interview-message ${role}`;
        const meta = document.createElement("span");
        meta.textContent = label;
        const body = document.createElement("p");
        body.textContent = text;
        message.append(meta, body);
        conversationList.appendChild(message);
        conversationList.scrollTop = conversationList.scrollHeight;
    }

    function updateRound(round, total) {
        const safeTotal = Math.max(total || 1, 1);
        const safeRound = Math.min(Math.max(round || 1, 1), safeTotal);
        roundLabel.textContent = `第 ${safeRound} / ${safeTotal} 轮`;
        roundProgress.style.width = `${safeRound / safeTotal * 100}%`;
    }

    function renderAdvice(feedback) {
        const scores = [
            ["综合表现", feedback.overall_score],
            ["结构", feedback.structure_score],
            ["相关度", feedback.relevance_score],
            ["证据", feedback.evidence_score],
            ["表达", feedback.expression_score]
        ];
        const strengths = (feedback.strengths || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
        const suggestions = (feedback.suggestions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
        adviceContent.className = "";
        adviceContent.innerHTML = `
            <p class="interview-feedback-text">${escapeHtml(feedback.feedback_text)}</p>
            <div class="interview-score-strip">
                ${scores.map(([name, value]) => `<div><strong>${Number(value || 0)}</strong><span>${escapeHtml(name)}</span></div>`).join("")}
            </div>
            <h3>亮点</h3><ul>${strengths}</ul>
            <h3>建议</h3><ul>${suggestions}</ul>
            <h3>参考补强</h3><p class="interview-polished">${escapeHtml(feedback.polished_answer)}</p>
        `;
    }

    function renderFinalReport(report) {
        const finalPanel = document.createElement("article");
        finalPanel.className = "interview-message interviewer final";
        const improvements = (report?.improvements || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
        const nextActions = (report?.next_actions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
        finalPanel.innerHTML = `
            <span>面试总结</span>
            <p>${escapeHtml(report?.summary)} 本次平均表现 ${Number(report?.average_score || 0)} 分。</p>
            <div class="interview-final-lists">
                <div><strong>继续优化</strong><ul>${improvements}</ul></div>
                <div><strong>下一步</strong><ul>${nextActions}</ul></div>
            </div>
        `;
        conversationList.appendChild(finalPanel);
        conversationList.scrollTop = conversationList.scrollHeight;
    }

    function showCameraFallback(message) {
        candidateVideo.hidden = true;
        cameraPlaceholder.hidden = false;
        cameraPlaceholder.innerHTML = `
            <span></span><strong>摄像头暂不可用</strong>
            <p>${escapeHtml(message)}</p>
            <button type="button" id="retryCameraButton">重新尝试</button>
        `;
        cameraBadge.textContent = "可继续文字面试";
        document.getElementById("retryCameraButton").addEventListener("click", startCameraAndExpressionMonitor);
        expressionStatus.textContent = "等待摄像头";
        expressionLabel.textContent = "未启用";
    }

    function loadFaceApi() {
        if (window.faceapi) {
            return Promise.resolve(window.faceapi);
        }
        return new Promise((resolve, reject) => {
            const script = document.createElement("script");
            const timer = window.setTimeout(() => reject(new Error("表情模型加载超时")), 12000);
            script.src = FACE_API_SCRIPT;
            script.async = true;
            script.onload = () => {
                window.clearTimeout(timer);
                window.faceapi ? resolve(window.faceapi) : reject(new Error("表情模型初始化失败"));
            };
            script.onerror = () => {
                window.clearTimeout(timer);
                reject(new Error("表情模型资源不可用"));
            };
            document.head.appendChild(script);
        });
    }

    function prepareFaceModels() {
        if (!faceModelPromise) {
            faceModelPromise = loadFaceApi().then(async (faceapi) => {
                await Promise.all([
                    faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_URL),
                    faceapi.nets.faceExpressionNet.loadFromUri(FACE_MODEL_URL)
                ]);
                return faceapi;
            }).catch((error) => {
                faceModelPromise = null;
                throw error;
            });
        }
        return faceModelPromise;
    }

    function updateExpressionUi(expressions) {
        const entries = Object.entries(expressions || {}).sort((left, right) => right[1] - left[1]);
        const dominant = entries[0]?.[0] || "neutral";
        const rawTension = Math.min(100, Math.round(
            (expressions.fearful || 0) * 85 +
            (expressions.angry || 0) * 55 +
            (expressions.sad || 0) * 35 +
            (expressions.disgusted || 0) * 30 +
            (expressions.surprised || 0) * 18
        ));
        smoothedTension = Math.round(smoothedTension * 0.68 + rawTension * 0.32);
        expressionLabel.textContent = expressionNames[dominant] || "自然专注";
        tensionValue.textContent = `${smoothedTension}%`;
        tensionBar.style.width = `${smoothedTension}%`;
        tensionBar.classList.toggle("high", smoothedTension >= 35);

        tenseStreak = smoothedTension >= 35 ? tenseStreak + 1 : Math.max(0, tenseStreak - 1);
        const now = Date.now();
        if (tenseStreak >= 4 && now - lastReminderAt > 20000) {
            lastReminderAt = now;
            calmReminder.classList.add("active");
            calmReminder.querySelector("p").textContent = "检测到持续紧张趋势。先慢慢呼气，放松肩膀，停顿两秒后再继续回答。";
            window.setTimeout(() => calmReminder.classList.remove("active"), 8000);
        }
    }

    async function startExpressionMonitor() {
        expressionStatus.textContent = "模型加载中";
        try {
            const faceapi = await prepareFaceModels();
            if (expressionTimer) {
                window.clearInterval(expressionTimer);
            }
            faceMissCount = 0;
            expressionStatus.textContent = "本地实时分析";
            expressionTimer = window.setInterval(async () => {
                if (expressionBusy || candidateVideo.readyState < 2) {
                    return;
                }
                expressionBusy = true;
                try {
                    const expandedSearch = faceMissCount >= 2;
                    const detection = await faceapi
                        .detectSingleFace(candidateVideo, new faceapi.TinyFaceDetectorOptions({
                            inputSize: expandedSearch ? 416 : 320,
                            scoreThreshold: expandedSearch ? 0.12 : 0.2
                        }))
                        .withFaceExpressions();
                    if (!detection) {
                        faceMissCount += 1;
                        expressionStatus.textContent = faceMissCount >= 3 ? "正在扩大检测范围" : "正在定位面部";
                        expressionLabel.textContent = faceMissCount >= 3 ? "请靠近镜头" : "正在识别";
                        tensionValue.textContent = "--";
                        tensionBar.style.width = "0%";
                        if (faceMissCount >= 3) {
                            calmReminder.querySelector("p").textContent = "请稍微靠近镜头，让面部占画面约三分之一，并避免背光。";
                        }
                    } else {
                        if (faceMissCount >= 3) {
                            calmReminder.querySelector("p").textContent = "已检测到面部。保持自然呼吸，看向镜头，按自己的节奏回答。";
                        }
                        faceMissCount = 0;
                        expressionStatus.textContent = "本地实时分析";
                        updateExpressionUi(detection.expressions);
                    }
                } catch (_error) {
                    expressionStatus.textContent = "分析暂时中断";
                } finally {
                    expressionBusy = false;
                }
            }, 900);
        } catch (_error) {
            expressionStatus.textContent = "高级模型不可用";
            expressionLabel.textContent = "摄像头已开启";
            tensionValue.textContent = "--";
        }
    }

    async function startCameraAndExpressionMonitor() {
        if (!navigator.mediaDevices?.getUserMedia) {
            showCameraFallback("当前浏览器不支持摄像头访问，请使用最新版 Chrome 或 Edge。");
            return;
        }
        if (cameraStream) {
            cameraStream.getTracks().forEach((track) => track.stop());
        }
        cameraPlaceholder.hidden = false;
        cameraBadge.textContent = "正在申请权限";
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 540 } },
                audio: false
            });
            candidateVideo.srcObject = cameraStream;
            candidateVideo.hidden = false;
            await candidateVideo.play();
            cameraPlaceholder.hidden = true;
            cameraBadge.textContent = `本地画面 · ${candidateVideo.videoWidth || 0}×${candidateVideo.videoHeight || 0}`;
            startExpressionMonitor();
        } catch (error) {
            const denied = error?.name === "NotAllowedError";
            showCameraFallback(denied ? "未获得摄像头权限。你可以授权后重试，或继续文字面试。" : "没有检测到可用摄像头，可继续文字面试。");
        }
    }

    startVideoButton.addEventListener("click", () => {
        if (!interviewSession) {
            return;
        }
        writtenResultPhase.hidden = true;
        videoPhase.hidden = false;
        roleLabel.textContent = interviewSession.target_role || "目标岗位";
        updateRound(interviewSession.current_round, interviewSession.total_rounds);
        appendMessage("interviewer", startVideoButton.dataset.openingMessage, interviewSession.interviewer_name);
        appendMessage("interviewer", startVideoButton.dataset.firstQuestion, `第 ${interviewSession.current_round} 轮问题`);
        setStep("video");
        window.scrollTo({ top: 0, behavior: "smooth" });
        answerText.focus();
    });

    enableCameraButton.addEventListener("click", startCameraAndExpressionMonitor);

    window.setTimeout(() => {
        prepareFaceModels().catch(() => {
            expressionStatus.textContent = "等待摄像头";
        });
    }, 800);

    answerForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!interviewSession) {
            return;
        }
        const answer = answerText.value.trim();
        if (!answer) {
            appendMessage("interviewer", "请先完整回答当前问题，我再给你点评。", "提醒");
            return;
        }

        const currentQuestion = interviewSession.current_question;
        const currentRound = interviewSession.current_round;
        appendMessage("candidate", answer, "我的回答");
        answerText.value = "";
        answerText.disabled = true;
        submitAnswerButton.disabled = true;
        submitAnswerButton.textContent = "点评中...";
        try {
            const response = await fetch("/interview/answer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session: interviewSession,
                    question: currentQuestion,
                    answer,
                    round_index: currentRound,
                    history: interviewHistory
                })
            });
            const data = await response.json();
            if (!data.ok) {
                throw new Error((data.errors || ["提交失败，请重试。"]).join(" "));
            }

            interviewHistory.push(data.history_item);
            renderAdvice(data.feedback);
            interviewSession = data.session;
            if (data.finished) {
                updateRound(data.total_rounds, data.total_rounds);
                appendMessage("interviewer", "本次视频面试结束。下面是你的整体复盘。", interviewSession.interviewer_name);
                renderFinalReport(data.final_report);
                submitAnswerButton.textContent = "面试已完成";
                setStep("review");
                if (expressionTimer) {
                    window.clearInterval(expressionTimer);
                }
                if (cameraStream) {
                    cameraStream.getTracks().forEach((track) => track.stop());
                }
                cameraBadge.textContent = "面试已完成";
                return;
            }

            updateRound(data.next_round, data.total_rounds);
            appendMessage("interviewer", data.next_question, `第 ${data.next_round} 轮问题`);
            answerText.disabled = false;
            submitAnswerButton.disabled = false;
            submitAnswerButton.textContent = "提交回答";
            answerText.focus();
        } catch (error) {
            appendMessage("interviewer", error.message || "提交失败，请检查服务是否正在运行。", "提醒");
            answerText.disabled = false;
            submitAnswerButton.disabled = false;
            submitAnswerButton.textContent = "提交回答";
        }
    });

    window.addEventListener("beforeunload", () => {
        if (expressionTimer) {
            window.clearInterval(expressionTimer);
        }
        if (cameraStream) {
            cameraStream.getTracks().forEach((track) => track.stop());
        }
    });
})();
