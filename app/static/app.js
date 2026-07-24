const form = document.querySelector("#predict-form");
const label = document.querySelector("#input-label");
const result = document.querySelector("#result");
const tabs = document.querySelectorAll(".tab");

let mode = "text";

tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        mode = tab.dataset.mode;

        tabs.forEach((item) => {
            const isActive = item === tab;
            item.classList.toggle("active", isActive);
            item.setAttribute("aria-selected", String(isActive));
        });

        result.classList.add("hidden");
        const activeInput = document.querySelector("#news-input");

        if (mode === "url") {
            label.textContent = "News URL";
            activeInput.outerHTML = '<input id="news-input" type="url" placeholder="https://example.com/news/article" required>';
        } else {
            label.textContent = "Article text";
            activeInput.outerHTML = '<textarea id="news-input" rows="10" placeholder="Paste news text here..." required></textarea>';
        }
    });
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const activeInput = document.querySelector("#news-input");
    const button = form.querySelector("button");
    const value = activeInput.value.trim();

    if (!value) {
        return;
    }

    button.disabled = true;
    button.textContent = "Analyzing...";
    result.classList.remove("hidden");
    result.innerHTML = "<h2>Checking article...</h2>";

    try {
        const endpoint = mode === "url" ? "/predict_url" : "/predict";
        const payload = mode === "url" ? { url: value } : { text: value };

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

        renderResult(data);
    } catch (error) {
        result.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
    } finally {
        button.disabled = false;
        button.textContent = "Analyze";
    }
});

function renderResult(data) {
    const results = [
        data.ml_result || {
            type: "ML",
            model: "SVM + TF-IDF",
            prediction: data.prediction,
            confidence: data.confidence
        },
        data.dl_result
    ].filter(Boolean);

    const details = [
        ["Language", data.language],
        ["Title", data.title],
        ["Source reputation", data.source_reputation],
        ["Translated text", data.translated_text]
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");

    result.innerHTML = `
        <h2>Separate Model Results</h2>
        <div class="model-results">
            ${results.map(renderModelResult).join("")}
        </div>
        <div class="details">
            ${details.map(([name, value]) => `
                <div class="detail">
                    <span>${escapeHtml(name)}</span>
                    <span>${escapeHtml(String(value))}</span>
                </div>
            `).join("")}
        </div>
    `;
}

function renderModelResult(modelResult) {
    const prediction = String(modelResult.prediction || "Unavailable");
    const normalized = prediction.toLowerCase();
    const badgeClass = modelResult.error
        ? "unavailable"
        : normalized.includes("real") || normalized.includes("true")
            ? "real"
            : "fake";
    const confidence = modelResult.confidence !== undefined ? formatConfidence(modelResult.confidence) : "Not available";
    const error = modelResult.error ? `<p class="model-error">${escapeHtml(modelResult.error)}</p>` : "";

    return `
        <article class="model-card">
            <div>
                <span class="model-type">${escapeHtml(modelResult.type || "Model")}</span>
                <h3>${escapeHtml(modelResult.model || "Unknown model")}</h3>
            </div>
            <span class="badge ${badgeClass}">${escapeHtml(prediction)}</span>
            <div class="model-confidence">
                <span>Confidence</span>
                <strong>${escapeHtml(confidence)}</strong>
            </div>
            ${error}
        </article>
    `;
}

function formatConfidence(value) {
    const number = Number(value);

    if (Number.isNaN(number)) {
        return String(value);
    }

    return number <= 1 ? `${Math.round(number * 100)}%` : `${Math.round(number)}%`;
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
