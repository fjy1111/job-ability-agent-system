(function () {
    "use strict";

    const container = document.getElementById("radarChart");
    const canvas = document.getElementById("radarCanvas");

    if (!container || !canvas) {
        return;
    }

    const labels = ["专业基础", "职业发展", "工具技能", "技术实践"];
    let scores = [
        Number(container.dataset.professional),
        Number(container.dataset.career),
        Number(container.dataset.tools),
        Number(container.dataset.practice),
    ].map(function (score) {
        return Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 0;
    });

    const context = canvas.getContext("2d");

    function pointAt(centerX, centerY, radius, index, scale) {
        const angle = -Math.PI / 2 + index * Math.PI / 2;
        return {
            x: centerX + Math.cos(angle) * radius * scale,
            y: centerY + Math.sin(angle) * radius * scale,
        };
    }

    function drawPolygon(points, fillStyle, strokeStyle, lineWidth) {
        context.beginPath();
        points.forEach(function (point, index) {
            if (index === 0) {
                context.moveTo(point.x, point.y);
            } else {
                context.lineTo(point.x, point.y);
            }
        });
        context.closePath();

        if (fillStyle) {
            context.fillStyle = fillStyle;
            context.fill();
        }

        if (strokeStyle) {
            context.strokeStyle = strokeStyle;
            context.lineWidth = lineWidth;
            context.stroke();
        }
    }

    function draw() {
        const width = Math.max(container.clientWidth, 320);
        const height = Math.max(container.clientHeight, 320);
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

        canvas.width = Math.round(width * pixelRatio);
        canvas.height = Math.round(height * pixelRatio);
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        context.clearRect(0, 0, width, height);

        const centerX = width / 2;
        const centerY = height / 2 + 4;
        const radius = Math.min(width * 0.29, height * 0.34);

        for (let level = 5; level >= 1; level -= 1) {
            const scale = level / 5;
            const gridPoints = labels.map(function (_, index) {
                return pointAt(centerX, centerY, radius, index, scale);
            });
            drawPolygon(
                gridPoints,
                level % 2 === 0 ? "#f8fafc" : "#ffffff",
                "#dbe3ef",
                1
            );
        }

        context.strokeStyle = "#dbe3ef";
        context.lineWidth = 1;
        labels.forEach(function (_, index) {
            const point = pointAt(centerX, centerY, radius, index, 1);
            context.beginPath();
            context.moveTo(centerX, centerY);
            context.lineTo(point.x, point.y);
            context.stroke();
        });

        const scorePoints = scores.map(function (score, index) {
            return pointAt(centerX, centerY, radius, index, score / 100);
        });
        drawPolygon(scorePoints, "rgba(37, 99, 235, 0.18)", "#2563eb", 3);

        context.fillStyle = "#2563eb";
        scorePoints.forEach(function (point) {
            context.beginPath();
            context.arc(point.x, point.y, 4, 0, Math.PI * 2);
            context.fill();
        });

        context.fillStyle = "#475569";
        context.font = "700 14px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        context.textBaseline = "middle";

        labels.forEach(function (label, index) {
            const point = pointAt(centerX, centerY, radius + 28, index, 1);
            context.textAlign = index === 1 ? "left" : index === 3 ? "right" : "center";
            context.fillText(label, point.x, point.y);
        });
    }

    draw();

    window.updateAbilityRadar = function (nextScores) {
        const values = nextScores || {};
        scores = [
            Number(values.professional),
            Number(values.career),
            Number(values.tools),
            Number(values.practice),
        ].map(function (score) {
            return Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 0;
        });
        container.dataset.professional = String(scores[0]);
        container.dataset.career = String(scores[1]);
        container.dataset.tools = String(scores[2]);
        container.dataset.practice = String(scores[3]);
        draw();
    };

    if ("ResizeObserver" in window) {
        const resizeObserver = new ResizeObserver(draw);
        resizeObserver.observe(container);
    } else {
        window.addEventListener("resize", draw);
    }
})();
