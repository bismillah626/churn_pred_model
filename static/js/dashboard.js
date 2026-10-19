/* Client-side Power BI-style slicers and Plotly redraws. */
(function () {
    "use strict";

    const state = { payload: null, filteredRows: [], search: "" };
    const $ = (selector) => document.querySelector(selector);
    const $$ = (selector) => Array.from(document.querySelectorAll(selector));

    function css(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    function layout(title, xTitle, yTitle, extra) {
        return Object.assign({
            title: { text: title, font: { family: "Space Grotesk", size: 15 } },
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { family: "DM Sans", color: css("--text"), size: 11 },
            margin: { l: 52, r: 18, t: 44, b: 52 },
            hoverlabel: { bgcolor: css("--surface"), font: { color: css("--text") } },
            legend: { orientation: "h", y: -0.2, font: { size: 10 } },
            xaxis: { title: xTitle, gridcolor: css("--line"), zerolinecolor: css("--line") },
            yaxis: { title: yTitle, gridcolor: css("--line"), zerolinecolor: css("--line") },
        }, extra || {});
    }

    function draw(id, traces, chartLayout) {
        const element = document.getElementById(id);
        if (!element || !window.Plotly) return;
        Plotly.react(element, traces, chartLayout, { responsive: true, displayModeBar: false });
    }

    function setCaption(id, text) {
        const element = document.getElementById(id);
        if (element) element.textContent = text;
    }

    function number(value) {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : 0;
    }

    function percent(value) {
        return `${(number(value) * 100).toFixed(1)}%`;
    }

    function filteredRows() {
        if (!state.payload) return [];
        const geography = $("#filter-geography")?.value || "All";
        const gender = $("#filter-gender")?.value || "All";
        const active = $("#filter-active")?.value || "All";
        const product = $("#filter-product")?.value || "All";
        const minAge = number($("#filter-age-min")?.value);
        const maxAge = number($("#filter-age-max")?.value);
        return state.payload.rows.filter((row) => {
            return (geography === "All" || row.Geography === geography) &&
                (gender === "All" || row.Gender === gender) &&
                (active === "All" || String(row.IsActiveMember) === active) &&
                (product === "All" || row.NumOfProducts === product) &&
                number(row.Age) >= minAge && number(row.Age) <= maxAge;
        });
    }

    function populateSelect(id, values, labels) {
        const select = document.getElementById(id);
        if (!select) return;
        select.innerHTML = `<option value="All">All</option>` + values.map((value) => {
            const label = labels?.[value] || value;
            return `<option value="${String(value).replaceAll('"', '&quot;')}">${label}</option>`;
        }).join("");
    }

    function updateKpis(rows) {
        const metadata = state.payload.metadata;
        const churnRate = rows.length ? rows.reduce((sum, row) => sum + number(row.Exited), 0) / rows.length : 0;
        $("#kpi-row-count").textContent = rows.length.toLocaleString();
        $("#kpi-churn-rate").textContent = percent(churnRate);
        $("#kpi-credit-score").textContent = rows.length ? (rows.reduce((sum, row) => sum + number(row.CreditScore), 0) / rows.length).toFixed(0) : "—";
        $("#kpi-best-model").textContent = state.payload.model.best_model;
        $("#filter-summary").textContent = `${rows.length.toLocaleString()} of ${metadata.row_count.toLocaleString()} customers visible`;
    }

    function renderTarget(rows) {
        const churned = rows.reduce((sum, row) => sum + (number(row.Exited) === 1 ? 1 : 0), 0);
        draw("target-chart", [{
            labels: ["Stayed", "Churned"], values: [rows.length - churned, churned], type: "pie",
            hole: 0.62, marker: { colors: [css("--teal"), css("--danger")] },
            textinfo: "label+percent", hovertemplate: "%{label}: %{value} customers<extra></extra>",
        }], { ...layout("Target distribution", "", "", { margin: { l: 12, r: 12, t: 44, b: 20 }, showlegend: true }), showlegend: true });
        setCaption("target-insight", rows.length ? `${percent(churned / rows.length)} of the filtered customers are predicted as the positive churn outcome in the observed data.` : "No customers match the current slicers.");
    }

    function renderDistributions(rows) {
        const age = rows.map((row) => number(row.Age));
        const credit = rows.map((row) => number(row.CreditScore));
        draw("histogram-chart", [
            { x: age, type: "histogram", name: "Age", opacity: .68, marker: { color: css("--primary") }, hovertemplate: "Age: %{x}<br>Customers: %{y}<extra></extra>" },
            { x: credit, type: "histogram", name: "Credit score", opacity: .55, marker: { color: css("--teal") }, hovertemplate: "Credit score: %{x}<br>Customers: %{y}<extra></extra>" },
        ], layout("Feature distributions", "Feature value", "Customers", { barmode: "overlay" }));
        draw("boxplot-chart", ["Age", "CreditScore", "Balance", "EstimatedSalary"].map((field, index) => ({
            y: rows.map((row) => number(row[field])), x: rows.map(() => field), type: "box", name: field,
            marker: { color: [css("--primary"), css("--teal"), css("--warning"), css("--danger")][index] },
            boxpoints: false, hovertemplate: `${field}: %{y:.2f}<extra></extra>`,
        })), layout("Feature spread and outliers", "Feature", "Value", { showlegend: true }));
        setCaption("distribution-insight", rows.length ? `Age is concentrated around ${(age.reduce((a, b) => a + b, 0) / age.length).toFixed(0)} years; the score and balance boxes reveal the cohort's spread and outliers.` : "No data for this filter.");
        setCaption("boxplot-insight", rows.length ? "The box plots compare the center, spread, and outliers of the most useful numeric customer features." : "No data for this filter.");
    }

    function renderCorrelation(rows) {
        const fields = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary", "Exited"];
        const matrix = fields.map((fieldA) => fields.map((fieldB) => {
            const a = rows.map((row) => number(row[fieldA]));
            const b = rows.map((row) => number(row[fieldB]));
            if (a.length < 2) return 0;
            const meanA = a.reduce((x, y) => x + y, 0) / a.length;
            const meanB = b.reduce((x, y) => x + y, 0) / b.length;
            const numerator = a.reduce((sum, value, i) => sum + (value - meanA) * (b[i] - meanB), 0);
            const denominator = Math.sqrt(a.reduce((sum, value) => sum + (value - meanA) ** 2, 0) * b.reduce((sum, value) => sum + (value - meanB) ** 2, 0));
            return denominator ? numerator / denominator : 0;
        }));
        draw("correlation-chart", [{
            z: matrix, x: fields, y: fields, type: "heatmap", zmin: -1, zmax: 1,
            colorscale: [[0, "#e35b70"], [.5, "#f7f8fc"], [1, "#5546e8"]],
            hovertemplate: "%{y} × %{x}: %{z:.2f}<extra></extra>", colorbar: { title: "r" },
        }], layout("Correlation heatmap", "Feature", "Feature", { margin: { l: 105, r: 20, t: 44, b: 80 } }));
        setCaption("correlation-insight", "The heatmap uses Pearson correlation; darker cells indicate stronger positive or negative linear relationships in the filtered cohort.");
    }

    function renderCategories(rows) {
        const groups = [];
        ["Geography", "Gender", "NumOfProducts"].forEach((field) => {
            [...new Set(rows.map((row) => row[field]))].sort().forEach((value) => {
                const group = rows.filter((row) => row[field] === value);
                groups.push({ label: `${field}: ${value}`, stayed: group.filter((row) => number(row.Exited) === 0).length, churned: group.filter((row) => number(row.Exited) === 1).length });
            });
        });
        draw("category-chart", [
            { x: groups.map((group) => group.label), y: groups.map((group) => group.stayed), name: "Stayed", type: "bar", marker: { color: css("--teal") }, hovertemplate: "%{x}<br>Stayed: %{y}<extra></extra>" },
            { x: groups.map((group) => group.label), y: groups.map((group) => group.churned), name: "Churned", type: "bar", marker: { color: css("--danger") }, hovertemplate: "%{x}<br>Churned: %{y}<extra></extra>" },
        ], layout("Categorical features vs target", "Category", "Customers", { barmode: "group", xaxis: { tickangle: -28, title: "Category" } }));
        const highest = groups.sort((a, b) => (b.churned / Math.max(b.stayed + b.churned, 1)) - (a.churned / Math.max(a.stayed + a.churned, 1)))[0];
        setCaption("category-insight", highest ? `${highest.label} has the highest observed churn share among the visible category groups (${percent(highest.churned / (highest.stayed + highest.churned))}).` : "No category data for this filter.");
    }

    function renderModelComparison() {
        const metrics = state.payload.model.model_comparison;
        const names = metrics.map((item) => item.model);
        draw("model-chart", ["accuracy", "precision", "recall", "f1", "roc_auc"].map((metric, index) => ({
            x: names, y: metrics.map((item) => item[metric]), name: metric === "roc_auc" ? "ROC-AUC" : metric.toUpperCase(), type: "bar",
            marker: { color: [css("--primary"), css("--teal"), css("--warning"), css("--danger"), "#8b78d9"][index] },
            hovertemplate: `${metric}: %{y:.3f}<extra></extra>`,
        })), layout("Model comparison", "Model", "Score", { barmode: "group", yaxis: { title: "Score", range: [0, 1] }, xaxis: { title: "Model", tickangle: -20 } }));
        const best = metrics.find((item) => item.model === state.payload.model.best_model) || metrics[0];
        setCaption("model-insight", `${best.model} leads the holdout comparison on accuracy (${percent(best.accuracy)}); use recall and ROC-AUC to balance the cost of missed churners.`);
    }

    function renderCurves() {
        const model = state.payload.model;
        draw("roc-chart", model.roc_curves.map((curve) => ({ x: curve.fpr, y: curve.tpr, type: "scatter", mode: "lines", name: curve.model, hovertemplate: "False positive: %{x:.2f}<br>True positive: %{y:.2f}<extra>%{fullData.name}</extra>" })), layout("ROC curves", "False positive rate", "True positive rate", { xaxis: { title: "False positive rate", range: [0, 1] }, yaxis: { title: "True positive rate", range: [0, 1] } }));
        draw("pr-chart", model.pr_curves.map((curve) => ({ x: curve.recall, y: curve.precision, type: "scatter", mode: "lines", name: curve.model, hovertemplate: "Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra>%{fullData.name}</extra>" })), layout("Precision–recall curves", "Recall", "Precision", { xaxis: { title: "Recall", range: [0, 1] }, yaxis: { title: "Precision", range: [0, 1] } }));
        setCaption("roc-insight", "A ROC curve closer to the top-left indicates better ranking quality across classification thresholds.");
        setCaption("pr-insight", "Precision–recall makes the trade-off visible for the less frequent churn class.");
        setCaption("curve-insight", "Curves are calculated on the same holdout sample so candidate models can be compared fairly.");
    }

    function renderConfusion() {
        const select = $("#confusion-select");
        if (!select) return;
        if (!select.options.length) {
            Object.keys(state.payload.model.confusion_matrices).forEach((name) => select.add(new Option(name, name)));
            select.value = state.payload.model.best_model;
        }
        const name = select.value || state.payload.model.best_model;
        const matrix = state.payload.model.confusion_matrices[name];
        draw("confusion-chart", [{
            z: matrix, x: ["Stayed", "Churned"], y: ["Stayed", "Churned"], type: "heatmap", colorscale: [[0, css("--surface-muted")], [1, css("--primary")]],
            text: matrix, texttemplate: "%{text}", hovertemplate: "Actual %{y}<br>Predicted %{x}: %{z} customers<extra></extra>", showscale: false,
        }], layout(`${name} confusion matrix`, "Predicted class", "Actual class", { margin: { l: 95, r: 20, t: 44, b: 65 } }));
        setCaption("confusion-insight", `This matrix contains ${state.payload.model.test_size.toLocaleString()} holdout customers; the upper-right cell is the missed-churn count to watch.`);
    }

    function renderImportance() {
        const model = state.payload.model;
        const featureRows = model.feature_importance.slice().reverse();
        draw("importance-chart", [{ x: featureRows.map((row) => row.importance), y: featureRows.map((row) => row.feature), type: "bar", name: "Importance", orientation: "h", marker: { color: css("--primary") }, hovertemplate: "%{y}<br>Importance: %{x:.4f}<extra></extra>" }], layout("Top 15 feature importance", "Relative importance", "Feature", { margin: { l: 135, r: 20, t: 44, b: 50 }, showlegend: true }));
        const shapRows = model.shap_summary.slice().reverse();
        draw("shap-chart", [{ x: shapRows.map((row) => row.importance), y: shapRows.map((row) => row.feature), type: "bar", name: model.shap_available ? "Mean absolute SHAP" : "Fallback importance", orientation: "h", marker: { color: css("--teal") }, hovertemplate: "%{y}<br>Mean absolute SHAP: %{x:.4f}<extra></extra>" }], layout(model.shap_available ? "SHAP summary · mean absolute impact" : "SHAP-style summary · fallback importance", "Mean absolute impact", "Feature", { margin: { l: 135, r: 20, t: 44, b: 50 }, showlegend: true }));
        setCaption("importance-insight", `The leading driver is ${model.feature_importance[0]?.feature || "not available"}; importance ranks drivers, not the direction of their effect.`);
        setCaption("shap-insight", model.shap_available ? "Mean absolute SHAP shows average contribution magnitude; it does not indicate whether a feature raises or lowers risk." : "SHAP was unavailable for this runtime, so the chart uses the model's feature-importance fallback.");
    }

    function renderProbabilities() {
        const probabilities = state.payload.model.probabilities;
        draw("probability-chart", [
            { x: probabilities.filter((row) => row.actual === 0).map((row) => row.probability), type: "histogram", name: "Stayed", opacity: .7, marker: { color: css("--teal") }, hovertemplate: "Probability: %{x:.2f}<br>Customers: %{y}<extra></extra>" },
            { x: probabilities.filter((row) => row.actual === 1).map((row) => row.probability), type: "histogram", name: "Churned", opacity: .7, marker: { color: css("--danger") }, hovertemplate: "Probability: %{x:.2f}<br>Customers: %{y}<extra></extra>" },
        ], layout("Prediction probability distribution", "Predicted churn probability", "Customers", { barmode: "overlay", xaxis: { title: "Predicted churn probability", range: [0, 1] } }));
        setCaption("probability-insight", `The best model's probability scores separate the two observed classes; the decision threshold defaults to 0.50.`);
    }

    function renderTable(rows) {
        const query = state.search.toLowerCase();
        const visible = rows.filter((row) => Object.values(row).join(" ").toLowerCase().includes(query)).slice(0, 50);
        const body = $("#data-table-body");
        if (!body) return;
        body.innerHTML = visible.map((row) => `<tr>
            <td>${row.CreditScore}</td><td>${row.Geography}</td><td>${row.Gender}</td><td>${row.Age}</td><td>${row.Tenure}</td>
            <td>${number(row.Balance).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td><td>${row.NumOfProducts}</td>
            <td>${row.IsActiveMember ? "Yes" : "No"}</td><td><span class="pill ${number(row.Exited) ? "yes" : "no"}">${number(row.Exited) ? "Churned" : "Stayed"}</span></td>
        </tr>`).join("") || `<tr><td colspan="9" class="empty-state">No rows match these filters.</td></tr>`;
        $("#table-summary").textContent = `Showing ${visible.length} of ${rows.length} filtered rows${query ? ` for “${state.search}”` : ""}.`;
    }

    function renderAll() {
        const rows = filteredRows();
        state.filteredRows = rows;
        updateKpis(rows);
        renderTarget(rows);
        renderDistributions(rows);
        renderCorrelation(rows);
        renderCategories(rows);
        renderModelComparison();
        renderCurves();
        renderConfusion();
        renderImportance();
        renderProbabilities();
        renderTable(rows);
    }

    function setupFilters() {
        const metadata = state.payload.metadata;
        populateSelect("filter-geography", metadata.geographies);
        populateSelect("filter-gender", metadata.genders);
        populateSelect("filter-product", metadata.products);
        populateSelect("filter-active", ["1", "0"], { "1": "Active", "0": "Inactive" });
        const ages = state.payload.rows.map((row) => number(row.Age));
        const min = Math.min(...ages), max = Math.max(...ages);
        ["filter-age-min", "filter-age-max"].forEach((id) => { const input = document.getElementById(id); input.min = min; input.max = max; input.value = id.endsWith("min") ? min : max; });
        $$(".slicer").forEach((input) => input.addEventListener("input", (event) => {
            const minInput = $("#filter-age-min"), maxInput = $("#filter-age-max");
            if (number(minInput.value) > number(maxInput.value)) {
                if (event.target === minInput) maxInput.value = minInput.value;
                else minInput.value = maxInput.value;
            }
            $("#age-range-value").textContent = `${minInput.value}–${maxInput.value}`;
            renderAll();
        }));
        ["filter-geography", "filter-gender", "filter-product", "filter-active", "confusion-select"].forEach((id) => document.getElementById(id)?.addEventListener("change", id === "confusion-select" ? renderConfusion : renderAll));
        $("#clear-filters")?.addEventListener("click", () => { ["filter-geography", "filter-gender", "filter-product", "filter-active"].forEach((id) => { const el = document.getElementById(id); if (el) el.value = "All"; }); $("#filter-age-min").value = min; $("#filter-age-max").value = max; $("#age-range-value").textContent = `${min}–${max}`; renderAll(); });
    }

    function setupTable() {
        $("#table-search")?.addEventListener("input", (event) => { state.search = event.target.value; renderTable(state.filteredRows); });
        $("#export-csv")?.addEventListener("click", () => {
            const rows = state.filteredRows;
            if (!rows.length) return;
            const columns = state.payload.columns;
            const csv = [columns.join(","), ...rows.map((row) => columns.map((column) => `"${String(row[column] ?? "").replaceAll('"', '""')}"`).join(","))].join("\n");
            const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); link.download = "filtered-churn-customers.csv"; link.click(); URL.revokeObjectURL(link.href);
        });
    }

    async function loadDashboard() {
        const loading = $("#dashboard-loading"), content = $("#dashboard-content"), error = $("#dashboard-error");
        try {
            if (!window.Plotly) throw new Error("Plotly could not be loaded. Check the browser network connection.");
            const response = await fetch("/api/dashboard");
            const payload = await response.json();
            if (!response.ok || payload.error) throw new Error(payload.error || "Dashboard data could not be loaded.");
            state.payload = payload;
            loading.classList.add("hidden"); content.classList.remove("hidden");
            setupFilters(); setupTable(); renderAll();
        } catch (dashboardError) {
            loading.classList.add("hidden"); error.classList.remove("hidden"); error.textContent = dashboardError.message;
        }
    }

    function setupPredictionForm() {
        const form = $("#prediction-form");
        if (!form) return;
        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const message = $("#prediction-message"), result = $("#prediction-result"), image = $("#prediction-explanation");
            message.textContent = "Scoring customer…";
            result.className = "prediction-result visible";
            try {
                const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(Object.fromEntries(new FormData(form))) });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload.error || "Prediction failed.");
                const churn = Number(payload.prediction) === 1;
                result.className = `prediction-result visible ${churn ? "churn" : "stay"}`;
                $("#prediction-label").textContent = churn ? "Customer likely to churn" : "Customer likely to stay";
                $("#prediction-detail").textContent = churn ? "Prioritize this customer for retention outreach." : "The model sees a lower immediate churn risk.";
                message.textContent = "Prediction complete";
                image.innerHTML = payload.shap_plot ? `<img src="data:image/png;base64,${payload.shap_plot}" alt="SHAP explanation for this prediction">` : `<p>SHAP explanation is not available in this runtime.</p>`;
            } catch (predictionError) {
                result.className = "prediction-result visible"; message.textContent = predictionError.message; image.innerHTML = "";
            }
        });
    }

    function setupChrome() {
        const root = document.documentElement;
        const saved = localStorage.getItem("churn-theme");
        if (saved) root.dataset.theme = saved;
        $("#theme-toggle")?.addEventListener("click", () => { root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark"; localStorage.setItem("churn-theme", root.dataset.theme); if (state.payload) renderAll(); });
        $("#menu-toggle")?.addEventListener("click", () => $("#sidebar")?.classList.toggle("open"));
    }

    document.addEventListener("DOMContentLoaded", () => { setupChrome(); setupPredictionForm(); if ($("#dashboard-app")) loadDashboard(); });
})();
