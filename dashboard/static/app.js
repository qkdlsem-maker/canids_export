"use strict";

const state = {
    scenario: "normal",
    scenarioData: null,
    running: false,
    runToken: 0,
    processed: 0,
    normal: 0,
    known: 0,
    unknown: 0,
    unsupported: 0,
    startTime: null,
};

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const KNOWN_CODES = new Set([1, 2, 3, 4]);

function fmt(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "—";
    }
    return Number(value).toFixed(digits);
}

function finalLabel(code) {
    const labels = {
        0: "NORMAL",
        1: "DoS",
        2: "FUZZY",
        3: "RPM",
        4: "GEAR",
        5: "UNKNOWN",
    };
    return labels[code] ?? `CODE ${code}`;
}

function resultClass(code) {
    if (code === 0) return "normal";
    if (code === 5) return "unknown";
    return "attack";
}

async function checkHealth() {
    try {
        const r = await fetch("/api/health");
        if (!r.ok) throw new Error("health failed");

        const data = await r.json();

        $("systemStatus").textContent = "SYSTEM ONLINE";
        $("statusDot").className = "status-dot online";
        $("mahaThreshold").textContent =
            fmt(data.mahalanobis_threshold, 3);
        $("pipelineThreshold").textContent =
            `Threshold ${fmt(data.mahalanobis_threshold, 3)}`;
    } catch (err) {
        $("systemStatus").textContent = "BACKEND OFFLINE";
        $("statusDot").className = "status-dot offline";
    }
}

async function loadScenario(name) {
    stopDemo();

    const r = await fetch(`/api/scenario/${name}`);
    if (!r.ok) throw new Error(`scenario load failed: ${name}`);

    const data = await r.json();

    state.scenario = name;
    state.scenarioData = data.scenario;

    document.querySelectorAll(".scenario").forEach(button => {
        button.classList.toggle(
            "active",
            button.dataset.scenario === name
        );
    });

    $("selectedScenario").textContent =
        state.scenarioData.display_name;

    $("scenarioSource").textContent =
        state.scenarioData.source;

    resetVisuals(false);
}

async function resetBackend() {
    const r = await fetch("/api/reset", {
        method: "POST",
    });

    if (!r.ok) {
        throw new Error("backend reset failed");
    }
}

function resetCounters() {
    state.processed = 0;
    state.normal = 0;
    state.known = 0;
    state.unknown = 0;
    state.unsupported = 0;
    state.startTime = null;

    updateCounters();
}

function resetVisuals(resetScenario = true) {
    resetCounters();

    $("trafficLog").innerHTML =
        '<div class="empty-log">CAN traffic will appear here.</div>';

    $("trafficState").textContent = "IDLE";

    updateProgress(0);

    $("currentCanId").textContent = "---";
    $("currentPayload").textContent = "----------------";

    $("lgbmClass").textContent = "—";
    $("lgbmProbability").textContent = "Probability —";

    $("pipelineDistance").textContent = "—";

    $("pipelineDecision").querySelector("strong").textContent =
        "READY";

    $("pipelineDecision").querySelector("small").textContent =
        "Hybrid";

    $("fFreq").textContent = "—";
    $("fUnique").textContent = "—";
    $("fEntropy").textContent = "—";
    $("fMean").textContent = "—";
    $("fDelta").textContent = "—";
    $("fValue").textContent = "—";

    setDecisionReady();

    if (resetScenario && state.scenarioData) {
        $("selectedScenario").textContent =
            state.scenarioData.display_name;
        $("scenarioSource").textContent =
            state.scenarioData.source;
    }
}

function setDecisionReady() {
    const hero = $("decisionHero");

    hero.className = "panel decision-hero normal";

    $("decisionIcon").textContent = "✓";
    $("finalDecision").textContent = "READY";
    $("decisionSubtext").textContent =
        "Select a scenario and run the demo";

    $("decisionStageTag").textContent = "WAITING";
    $("mahaDistance").textContent = "—";
}

function updateProgress(current = 0) {
    const el = $("trafficProgress");
    if (!el) return;

    const total =
        state.scenarioData && state.scenarioData.frames
            ? state.scenarioData.frames.length
            : 0;

    el.textContent = `${current} / ${total}`;
}

function updateCounters() {
    $("countProcessed").textContent =
        state.processed.toLocaleString();

    $("countNormal").textContent =
        state.normal.toLocaleString();

    $("countKnown").textContent =
        state.known.toLocaleString();

    $("countUnknown").textContent =
        state.unknown.toLocaleString();

    $("countUnsupported").textContent =
        state.unsupported.toLocaleString();

    let fps = 0;

    if (state.startTime && state.processed > 0) {
        const elapsed = (performance.now() - state.startTime) / 1000;
        if (elapsed > 0) {
            fps = state.processed / elapsed;
        }
    }

    $("demoRate").textContent = fps.toFixed(1);
}

function updateFeatures(features) {
    $("fFreq").textContent =
        fmt(features.freq_in_window, 4);

    $("fUnique").textContent =
        fmt(features.unique_ids_in_window, 0);

    $("fEntropy").textContent =
        fmt(features.entropy, 4);

    $("fMean").textContent =
        fmt(features.mean_byte, 3);

    $("fDelta").textContent =
        fmt(features.global_delta_zscore, 4);

    $("fValue").textContent =
        fmt(features.global_value_zscore, 4);
}

function updateDecision(prediction) {
    const code = prediction.final_code;
    const label = finalLabel(code);
    const cls = resultClass(code);
    const hero = $("decisionHero");

    hero.className = `panel decision-hero ${cls}`;

    if (code === 0) {
        $("decisionIcon").textContent = "✓";
        $("decisionSubtext").textContent =
            "Traffic classified as normal";
    } else if (code === 5) {
        $("decisionIcon").textContent = "?";
        $("decisionSubtext").textContent =
            "Unknown anomaly detected";
    } else {
        $("decisionIcon").textContent = "!";
        $("decisionSubtext").textContent =
            "Known attack detected";
    }

    $("finalDecision").textContent = label;
    $("decisionStageTag").textContent =
        prediction.decision_stage.toUpperCase();

    $("mahaDistance").textContent =
        fmt(prediction.mahalanobis_distance, 3);

    $("pipelineDistance").textContent =
        fmt(prediction.mahalanobis_distance, 3);

    $("lgbmClass").textContent =
        prediction.lightgbm_class;

    const bestProb = Math.max(...prediction.scores) * 100;

    $("lgbmProbability").textContent =
        `Probability ${bestProb.toFixed(1)}%`;

    const finalNode = $("pipelineDecision");

    finalNode.querySelector("strong").textContent = label;
    finalNode.querySelector("small").textContent =
        prediction.decision_stage;
}

function appendTraffic(frame, label, cls) {
    const log = $("trafficLog");

    const empty = log.querySelector(".empty-log");
    if (empty) empty.remove();

    const row = document.createElement("div");
    row.className = "traffic-row";

    row.innerHTML = `
        <span class="seq">#${String(state.processed).padStart(4, "0")}</span>
        <span class="can-id">${frame.can_id_hex}</span>
        <span class="payload">${frame.payload_hex}</span>
        <span class="result ${cls}">${label}</span>
    `;

    log.appendChild(row);

    while (log.children.length > 8) {
        log.removeChild(log.firstChild);
    }
}

async function processFrame(frame) {
    const canId = Number(frame.can_id);

    /*
     * Frozen MCU/F6 implementation supports standard 11-bit CAN.
     * Unsupported IDs are counted explicitly instead of silently
     * entering the detector.
     */
    if (canId < 0 || canId > 0x7FF) {
        state.unsupported += 1;
        state.processed += 1;

        $("currentCanId").textContent =
            frame.can_id_hex;

        $("currentPayload").textContent =
            frame.payload_hex;

        appendTraffic(
            frame,
            "UNSUPPORTED",
            "unsupported"
        );

        updateCounters();

        updateProgress(state.processed);

        return;
    }

    const r = await fetch("/api/frame", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            can_id: frame.can_id,
            payload: frame.payload_hex,
        }),
    });

    if (!r.ok) {
        throw new Error(`frame inference failed: ${r.status}`);
    }

    const data = await r.json();
    const prediction = data.prediction;
    const code = prediction.final_code;

    state.processed += 1;

    if (code === 0) {
        state.normal += 1;
    } else if (code === 5) {
        state.unknown += 1;
    } else if (KNOWN_CODES.has(code)) {
        state.known += 1;
    }

    $("currentCanId").textContent =
        data.frame.can_id_hex;

    $("currentPayload").textContent =
        data.frame.payload_hex;

    updateFeatures(data.features);
    updateDecision(prediction);

    appendTraffic(
        data.frame,
        finalLabel(code),
        resultClass(code)
    );

    updateCounters();

    if (state.scenarioData) {
        $("trafficProgress").textContent =
            `${state.processed} / ${state.scenarioData.frames.length}`;
    }
}

async function runDemo() {
    if (state.running || !state.scenarioData) return;

    state.running = true;
    state.runToken += 1;

    const token = state.runToken;

    $("runButton").textContent = "RUNNING...";
    $("trafficState").textContent = "STREAMING";

    try {
        await resetBackend();
        resetVisuals(false);

        state.startTime = performance.now();

        const frames = state.scenarioData.frames;

        updateProgress(0);

        for (let i = 0; i < frames.length; i++) {
            if (!state.running || token !== state.runToken) {
                break;
            }

            await processFrame(frames[i]);

            /*
             * Deliberately slowed for judge visibility.
             * This is visualization rate, not detector benchmark latency.
             */
            await sleep(35);
        }

        if (state.running && token === state.runToken) {
            $("trafficState").textContent = "COMPLETE";
        }

    } catch (err) {
        console.error(err);
        $("trafficState").textContent = "ERROR";
        $("systemStatus").textContent = "DEMO ERROR";
        $("statusDot").className = "status-dot offline";
    } finally {
        if (token === state.runToken) {
            state.running = false;
            $("runButton").textContent = "▶ RUN DEMO";
            updateCounters();
        }
    }
}

function stopDemo() {
    state.running = false;
    state.runToken += 1;

    if ($("runButton")) {
        $("runButton").textContent = "▶ RUN DEMO";
    }

    if ($("trafficState")) {
        $("trafficState").textContent = "STOPPED";
    }
}

async function fullReset() {
    stopDemo();

    try {
        await resetBackend();
    } catch (err) {
        console.error(err);
    }

    resetVisuals();
}

document.querySelectorAll(".scenario").forEach(button => {
    button.addEventListener("click", async () => {
        try {
            await loadScenario(button.dataset.scenario);
        } catch (err) {
            console.error(err);
            $("trafficState").textContent = "LOAD ERROR";
        }
    });
});

$("runButton").addEventListener("click", runDemo);
$("stopButton").addEventListener("click", stopDemo);
$("resetButton").addEventListener("click", fullReset);

(async function init() {
    await checkHealth();

    try {
        await loadScenario("normal");
    } catch (err) {
        console.error(err);
        $("trafficState").textContent = "LOAD ERROR";
    }
})();
