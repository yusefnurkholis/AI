const state = { accuracyChart: null, lossChart: null };

const $ = (selector) => document.querySelector(selector);
const formatPercent = (value) =>
    value === null || value === undefined || Number.isNaN(Number(value))
        ? "—"
        : `${(Number(value) * 100).toFixed(2)}%`;
const formatNumber = (value, digits = 4) =>
    value === null || value === undefined || Number.isNaN(Number(value))
        ? "—"
        : Number(value).toFixed(digits);

async function getJSON(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "Permintaan gagal.");
    return data;
}

async function loadStatus() {
    try {
        const status = await getJSON("/api/status");
        const modelStatus = $("#modelStatus");
        const dataStatus = $("#dataStatus");

        modelStatus.textContent = status.model_ready
            ? "● Model siap digunakan"
            : "○ Model belum disalin";
        modelStatus.classList.add(status.model_ready ? "ready" : "not-ready");

        const resultReady = status.history_ready || status.metrics_ready || status.report_ready;
        dataStatus.textContent = resultReady
            ? "● Hasil training tersedia"
            : "○ Hasil training belum tersedia";
        dataStatus.classList.add(resultReady ? "ready" : "not-ready");

        $("#architectureText").textContent = status.architecture || "EfficientNetB0";
        $("#imageSizeText").textContent = `${status.image_size || 224} × ${status.image_size || 224}`;

        if (!status.model_ready) {
            $("#predictButton").disabled = true;
            $("#predictMessage").textContent =
                "Salin file model ke folder model/ agar inference dapat digunakan.";
        }
    } catch (error) {
        $("#modelStatus").textContent = "Status server tidak tersedia";
        $("#dataStatus").textContent = "Status data tidak tersedia";
    }
}

function createLineChart(canvasId, labels, trainValues, validationValues, titleA, titleB) {
    const canvas = document.getElementById(canvasId);
    return new Chart(canvas, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: titleA,
                    data: trainValues,
                    borderColor: "#2f6f44",
                    backgroundColor: "rgba(47,111,68,.12)",
                    tension: .3,
                    fill: true,
                    pointRadius: 2
                },
                {
                    label: titleB,
                    data: validationValues,
                    borderColor: "#d7ad54",
                    backgroundColor: "rgba(215,173,84,.08)",
                    tension: .3,
                    fill: false,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: { legend: { position: "bottom" } },
            scales: {
                y: { beginAtZero: true, grid: { color: "rgba(0,0,0,.05)" } },
                x: { grid: { display: false } }
            }
        }
    });
}

function renderHistory(history) {
    const trainAcc = history.accuracy || [];
    const valAcc = history.val_accuracy || [];
    const trainLoss = history.loss || [];
    const valLoss = history.val_loss || [];
    const count = Math.max(trainAcc.length, valAcc.length, trainLoss.length, valLoss.length);
    const labels = Array.from({ length: count }, (_, i) => i + 1);

    if (!count) return false;

    if (state.accuracyChart) state.accuracyChart.destroy();
    if (state.lossChart) state.lossChart.destroy();

    state.accuracyChart = createLineChart(
        "accuracyChart", labels, trainAcc, valAcc, "Train Accuracy", "Validation Accuracy"
    );
    state.lossChart = createLineChart(
        "lossChart", labels, trainLoss, valLoss, "Train Loss", "Validation Loss"
    );
    return true;
}

function renderReport(rows) {
    const body = $("#reportBody");
    if (!rows || !rows.length) return false;

    const classKey = Object.keys(rows[0]).find(k => ["kelas", "", "Unnamed: 0"].includes(k)) || Object.keys(rows[0])[0];
    body.innerHTML = rows.map(row => {
        const className = row[classKey] ?? "—";
        const precision = row.precision;
        const recall = row.recall;
        const f1 = row["f1-score"];
        const support = row.support;
        return `<tr>
            <td>${className}</td>
            <td>${formatNumber(precision)}</td>
            <td>${formatNumber(recall)}</td>
            <td>${formatNumber(f1)}</td>
            <td>${support === undefined ? "—" : formatNumber(support, 0)}</td>
        </tr>`;
    }).join("");
    return true;
}

function renderMatrix(payload, fallbackLabels) {
    const host = $("#confusionMatrix");
    const matrix = payload.matrix || payload.values || [];
    const labels = payload.labels || fallbackLabels || [];
    if (!Array.isArray(matrix) || !matrix.length) return false;

    const maxValue = Math.max(...matrix.flat().map(Number), 1);
    const size = matrix.length + 1;
    const cells = [`<div class="matrix-cell matrix-label"></div>`];

    labels.forEach(label => cells.push(`<div class="matrix-cell matrix-label">${label}</div>`));
    matrix.forEach((row, rowIndex) => {
        cells.push(`<div class="matrix-cell matrix-label">${labels[rowIndex] || rowIndex}</div>`);
        row.forEach(value => {
            const intensity = .10 + (.85 * Number(value) / maxValue);
            cells.push(
                `<div class="matrix-cell" style="background:rgba(47,111,68,${intensity});color:${intensity > .55 ? "white" : "#18241a"}">${value}</div>`
            );
        });
    });

    host.innerHTML = `<div class="matrix-grid" style="grid-template-columns:repeat(${size},minmax(70px,1fr))">${cells.join("")}</div>`;
    return true;
}

async function loadResults() {
    try {
        const result = await getJSON("/api/results");
        const metrics = result.metrics || {};
        $("#metricAccuracy").textContent = formatPercent(metrics.test_accuracy);
        $("#metricLoss").textContent = formatNumber(metrics.test_loss);
        $("#metricPrecision").textContent = formatPercent(metrics.weighted_precision);
        $("#metricRecall").textContent = formatPercent(metrics.weighted_recall);
        $("#metricF1").textContent = formatPercent(metrics.weighted_f1_score);

        const hasHistory = renderHistory(result.history || {});
        const hasReport = renderReport(result.classification_report || []);
        const hasMatrix = renderMatrix(result.confusion_matrix || {}, result.class_names || []);
        const hasMetric = Object.values(metrics).some(v => v !== null && v !== undefined);

        if (!(hasHistory || hasReport || hasMatrix || hasMetric)) {
            $("#emptyResults").classList.remove("hidden");
        }
    } catch (error) {
        $("#emptyResults").classList.remove("hidden");
    }
}

function setupUpload() {
    const input = $("#imageInput");
    const preview = $("#imagePreview");
    const prompt = $("#uploadPrompt");
    const dropZone = $("#dropZone");

    function showFile(file) {
        if (!file) return;
        if (!file.type.startsWith("image/")) {
            $("#predictMessage").textContent = "Pilih file gambar yang valid.";
            $("#predictMessage").classList.add("error");
            return;
        }
        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.classList.remove("hidden");
        prompt.classList.add("hidden");
        $("#predictMessage").textContent = file.name;
        $("#predictMessage").classList.remove("error");
    }

    input.addEventListener("change", () => showFile(input.files[0]));
    ["dragenter", "dragover"].forEach(event =>
        dropZone.addEventListener(event, e => {
            e.preventDefault();
            dropZone.classList.add("dragging");
        })
    );
    ["dragleave", "drop"].forEach(event =>
        dropZone.addEventListener(event, e => {
            e.preventDefault();
            dropZone.classList.remove("dragging");
        })
    );
    dropZone.addEventListener("drop", e => {
        const files = e.dataTransfer.files;
        if (!files.length) return;
        const transfer = new DataTransfer();
        transfer.items.add(files[0]);
        input.files = transfer.files;
        showFile(files[0]);
    });
}

function renderPrediction(result) {
    $("#predictionEmpty").classList.add("hidden");
    $("#predictionContent").classList.remove("hidden");
    $("#predictedClass").textContent = result.predicted_class;
    $("#predictionConfidence").textContent = formatPercent(result.confidence);
    $("#confidenceBar").style.width = `${Math.min(100, result.confidence * 100)}%`;
    $("#uncertainWarning").classList.toggle("hidden", !result.uncertain);

    $("#probabilityList").innerHTML = result.probabilities.map(item => `
        <div class="probability-item">
            <div class="probability-head">
                <span>${item.class}</span>
                <strong>${formatPercent(item.probability)}</strong>
            </div>
            <div class="probability-bar">
                <span style="width:${Math.min(100, item.probability * 100)}%"></span>
            </div>
        </div>
    `).join("");
}

function setupPrediction() {
    $("#predictionForm").addEventListener("submit", async event => {
        event.preventDefault();
        const file = $("#imageInput").files[0];
        const message = $("#predictMessage");
        const button = $("#predictButton");

        if (!file) {
            message.textContent = "Pilih gambar terlebih dahulu.";
            message.classList.add("error");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        button.disabled = true;
        button.textContent = "Menganalisis...";
        message.textContent = "";
        message.classList.remove("error");

        try {
            const result = await getJSON("/api/predict", {
                method: "POST",
                body: formData
            });
            renderPrediction(result);
            message.textContent = "Analisis selesai.";
        } catch (error) {
            message.textContent = error.message;
            message.classList.add("error");
        } finally {
            button.disabled = false;
            button.textContent = "Analisis gambar";
        }
    });
}

function setupNavigation() {
    $("#menuButton").addEventListener("click", () => {
        document.querySelector("nav").classList.toggle("open");
    });
    document.querySelectorAll("nav a").forEach(link =>
        link.addEventListener("click", () => document.querySelector("nav").classList.remove("open"))
    );
}

document.addEventListener("DOMContentLoaded", () => {
    setupNavigation();
    setupUpload();
    setupPrediction();
    loadStatus();
    loadResults();
});
