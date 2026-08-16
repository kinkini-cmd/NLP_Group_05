const form = document.querySelector("#predict-form");
const label = document.querySelector("#input-label");
const result = document.querySelector("#result");
const tabs = document.querySelectorAll(".tab");
const inputCount = document.querySelector("#input-count");
const modelInputs = document.querySelectorAll('input[name="model_type"]');
const activeModelPill = document.querySelector("#active-model");

let mode = "text";

bindInputCounter();

tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        mode = tab.dataset.mode;

        tabs.forEach((item) => {
            const isActive = item === tab;
            item.classList.toggle("active", isActive);
            item.setAttribute("aria-selected", String(isActive));
        });

        resetResult();
        const activeInput = document.querySelector("#news-input");

        if (mode === "url") {
            label.textContent = "News URL";
            inputCount.textContent = "source scoring";
            activeInput.outerHTML = '<input id="news-input" type="url" placeholder="https://example.com/news/article" required>';
        } else {
            label.textContent = "Article text";
            activeInput.outerHTML = '<textarea id="news-input" rows="12" placeholder="Paste article text..." required></textarea>';
        }

        bindInputCounter();
        updateInputCounter();
    });
});

modelInputs.forEach((input) => {
    input.addEventListener("change", () => {
        updateActiveModelPill();
        resetResult();
    });
});

function updateActiveModelPill() {
    const selected = document.querySelector('input[name="model_type"]:checked');
    if (!selected) return;
    const map = { ml: "ML (SVM)", dl: "DL (CNN)", both: "Both" };
    activeModelPill.textContent = `${map[selected.value] || selected.value} active`;
}

updateActiveModelPill();

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const activeInput = document.querySelector("#news-input");
    const button = form.querySelector("button");
    const value = activeInput.value.trim();
    const selectedModel = document.querySelector('input[name="model_type"]:checked')?.value || "both";

    if (!value) {
        return;
    }

    button.disabled = true;
    button.textContent = "Running Prediction...";
    result.classList.remove("empty");
    result.innerHTML = `
        <div class="empty-state">
            <span class="empty-mark">...</span>
            <h2>Running Prediction</h2>
        </div>
    `;

    try {
        const endpoint = mode === "url" ? "/predict_url" : "/predict";
        const payload = {
            ...(mode === "url" ? { url: value } : { text: value }),
            model_type: selectedModel
        };

        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Prediction failed");
        }

        renderResult(data, selectedModel);
    } catch (error) {
        result.classList.remove("empty");
        result.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
    } finally {
        button.disabled = false;
        button.textContent = "Run Prediction";
    }
});

function bindInputCounter() {
    const activeInput = document.querySelector("#news-input");

    activeInput.addEventListener("input", updateInputCounter);
}

function updateInputCounter() {
    const activeInput = document.querySelector("#news-input");

    if (mode === "url") {
        inputCount.textContent = activeInput.value.trim() ? "source scoring" : "URL";
        return;
    }

    const words = getWordCount(activeInput.value);
    inputCount.textContent = `${words} ${words === 1 ? "word" : "words"}`;
}

function resetResult() {
    const selected = document.querySelector('input[name="model_type"]:checked');
    const map = { ml: "ML", dl: "DL", both: "ML" };
    const mark = map[selected?.value] || "ML";
    result.className = "result empty";
    result.innerHTML = `
        <div class="empty-state">
            <span class="empty-mark">${mark}</span>
            <h2>Waiting for input</h2>
        </div>
    `;
}

function renderResult(data, selectedModel) {
    let primaryResult = null;
    let secondaryResult = null;

    if (selectedModel === "ml" || !selectedModel) {
        primaryResult = data.ml_result || { type: "ML", model: "SVM + TF-IDF", prediction: data.prediction, confidence: data.confidence };
        secondaryResult = data.dl_result;
    } else if (selectedModel === "dl") {
        primaryResult = data.dl_result || { type: "DL", model: "XLM-RoBERTa", prediction: data.prediction, confidence: data.confidence };
        secondaryResult = data.ml_result;
    } else {
        primaryResult = data.ml_result || { type: "ML", model: "SVM + TF-IDF", prediction: data.prediction, confidence: data.confidence };
        secondaryResult = data.dl_result;
    }

    const prediction = String(primaryResult.prediction || "Unavailable");
    const normalized = prediction.toLowerCase();
    const badgeClass = normalized.includes("real") || normalized.includes("true")
        ? "real"
        : normalized.includes("fake") || normalized.includes("false")
            ? "fake"
            : "unavailable";
    const confidence = normalizeConfidence(primaryResult.confidence);
    const source = data.source_reputation || (mode === "url" ? "unknown" : "not used");
    const metrics = data.input_metrics || {};
    const notes = data.reliability_notes || [];
    const secondaryModel = renderSecondaryModel(secondaryResult);

    const signals = [
        ["Language", data.language || "unknown"],
        ["Words", metrics.word_count ?? getWordCount(data.translated_text || data.original_text || "")],
        ["Source", source],
        ["Mode", mode === "url" ? "URL" : "Text"]
    ];

    const details = [
        ["Title", data.title],
        ["Translated", data.translated_text && data.original_text && data.translated_text !== data.original_text ? data.translated_text : ""]
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");

    result.classList.remove("empty");
    result.innerHTML = `
        <article class="prediction-card">
            <div class="result-header">
                <div>
                    <h2 class="result-title">Prediction</h2>
                    <p class="model-name">${escapeHtml(primaryResult.model || "Selected Model")}</p>
                </div>
                <span class="badge ${badgeClass}">${escapeHtml(prediction)}</span>
            </div>

            <div class="confidence-block">
                <div class="metric-line">
                    <span>Confidence</span>
                    <strong>${confidence.label}</strong>
                </div>
                <div class="meter" aria-hidden="true">
                    <span class="meter-fill ${badgeClass}" style="width: ${confidence.value}%"></span>
                </div>
            </div>

            <div class="signal-grid">
                ${signals.map(([name, value]) => `
                    <div class="signal">
                        <span>${escapeHtml(name)}</span>
                        <strong>${escapeHtml(String(value))}</strong>
                    </div>
                `).join("")}
            </div>

            ${notes.length ? `
                <div class="notes">
                    ${notes.map((note) => `<p class="note">${escapeHtml(note)}</p>`).join("")}
                </div>
            ` : ""}

            ${details.length ? `
                <div class="details">
                    ${details.map(([name, value]) => `
                        <div class="detail">
                            <span>${escapeHtml(name)}</span>
                            <span>${escapeHtml(String(value))}</span>
                        </div>
                    `).join("")}
                </div>
            ` : ""}

            ${secondaryModel}
        </article>
    `;
}

function renderSecondaryModel(modelResult) {
    if (!modelResult || (!modelResult.error && !modelResult.prediction)) {
        return "";
    }

    if (modelResult.error) {
        return `
            <div class="secondary-model">
                <div class="metric-line">
                    <span>${escapeHtml(modelResult.model || "Secondary model")}</span>
                    <strong>Unavailable</strong>
                </div>
                <p class="model-error">${escapeHtml(modelResult.error)}</p>
            </div>
        `;
    }

    return `
        <div class="secondary-model">
            <div class="metric-line">
                <span>${escapeHtml(modelResult.model || "Secondary model")}</span>
                <strong>${escapeHtml(modelResult.prediction)} ${escapeHtml(normalizeConfidence(modelResult.confidence).label)}</strong>
            </div>
        </div>
    `;
}

function normalizeConfidence(value) {
    const number = Number(value);

    if (Number.isNaN(number)) {
        return {
            value: 0,
            label: "Not available"
        };
    }

    const percent = number <= 1 ? number * 100 : number;
    const bounded = Math.max(0, Math.min(percent, 100));

    return {
        value: bounded,
        label: `${bounded.toFixed(2).replace(/\.00$/, "")}%`
    };
}

function getWordCount(value) {
    return value.trim().split(/\s+/).filter(Boolean).length;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
