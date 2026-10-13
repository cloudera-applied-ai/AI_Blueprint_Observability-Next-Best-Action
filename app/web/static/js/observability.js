/**
 * The observability panel.
 *
 * This is the half of the demo a bank's product / risk / engineering teams
 * watch, so it gets a full column rather than the sidebar strip the Streamlit
 * build gave it. Everything here comes off the `TurnResponse` for the most
 * recent turn - the same fields visible on the `nba_turn` root run in
 * LangSmith. That is the point: what you see here is what an engineer would
 * see in the trace.
 *
 * Form choices, and why:
 *
 * - The risk probability is a single headline number, so it is a **hero
 *   figure** (one per view), not a chart.
 * - It is also a ratio against a limit, so it carries a **meter** with a tick
 *   at the guardrail threshold. A number with no reference point cannot tell
 *   you how close a customer came to being declined.
 * - Stage / clarify / guardrail are a handful of headline numbers, so they are
 *   a **KPI row** of stat tiles.
 *
 * Status is always a coloured dot *plus* a word, and the value text stays in
 * an ink token - meaning never rests on hue alone.
 */
import { langsmithThreadUrl, state } from "./state.js";
/** Human labels for `conversation_stage`, which is otherwise a bare enum. */
const STAGE_LABELS = {
    gathering: "Gathering information",
    offer_presented: "Offer presented",
    post_offer: "Post-offer chat",
};
function section(title) {
    const el = document.createElement("h3");
    el.className = "obs__section";
    el.textContent = title;
    return el;
}
function hint(text) {
    const el = document.createElement("p");
    el.className = "obs__hint";
    el.textContent = text;
    return el;
}
/**
 * Stat tile: label in sentence case with no trailing colon, then the value.
 * `status` adds a dot; the word in `value` is what actually carries the state.
 */
function tile(label, value, status = "neutral", wide = false) {
    const el = document.createElement("div");
    el.className = wide ? "tile tile--wide" : "tile";
    const labelEl = document.createElement("div");
    labelEl.className = "tile__label";
    labelEl.textContent = label;
    el.appendChild(labelEl);
    const valueEl = document.createElement("div");
    valueEl.className = "tile__value";
    if (status !== "neutral") {
        const dot = document.createElement("span");
        dot.className = `dot-mark dot-mark--${status}`;
        el.appendChild(dot);
        valueEl.prepend(dot);
    }
    valueEl.append(document.createTextNode(value));
    el.appendChild(valueEl);
    return el;
}
/**
 * Hero figure plus its meter.
 *
 * The fill turns to the breach colour once the probability crosses the
 * guardrail threshold, which is the same moment `risk_blocked` flips - so the
 * meter and the guardrail tile always agree.
 */
function riskHero(result) {
    const box = document.createElement("div");
    box.className = "obs__hero";
    const label = document.createElement("div");
    label.className = "obs__hero-label";
    label.textContent = "High-risk probability";
    box.appendChild(label);
    const probability = result.high_risk_probability;
    const value = document.createElement("div");
    value.className = "obs__hero-value";
    if (probability === null || probability === undefined) {
        value.classList.add("obs__hero-value--muted");
        value.textContent = "n/a";
        box.appendChild(value);
        box.appendChild(hint("Send a message to score the customer."));
        return box;
    }
    const pct = probability * 100;
    value.textContent = `${pct.toFixed(2)}%`;
    box.appendChild(value);
    const threshold = state.config?.high_risk_threshold ?? 0.5;
    const breached = probability >= threshold;
    const meter = document.createElement("div");
    meter.className = "meter";
    meter.setAttribute("role", "img");
    meter.setAttribute("aria-label", `High-risk probability ${pct.toFixed(2)} percent, guardrail threshold ${(threshold * 100).toFixed(0)} percent`);
    const fill = document.createElement("div");
    fill.className = breached ? "meter__fill meter__fill--breach" : "meter__fill";
    fill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    meter.appendChild(fill);
    const tick = document.createElement("div");
    tick.className = "meter__threshold";
    tick.style.left = `${Math.min(100, Math.max(0, threshold * 100))}%`;
    meter.appendChild(tick);
    box.appendChild(meter);
    const scale = document.createElement("div");
    scale.className = "meter__scale";
    const zero = document.createElement("span");
    zero.textContent = "0%";
    const mark = document.createElement("span");
    mark.textContent = `threshold ${(threshold * 100).toFixed(0)}%`;
    const hundred = document.createElement("span");
    hundred.textContent = "100%";
    scale.append(zero, mark, hundred);
    box.appendChild(scale);
    return box;
}
/** Discrete segments, so the cap is legible without reading the numbers. */
function clarifyTile(result) {
    const max = state.config?.max_clarify_turns ?? 3;
    const count = result.clarify_count ?? 0;
    const capped = count >= max;
    const el = tile("Clarify turns", capped ? `${count} / ${max} — cap reached` : `${count} / ${max}`, capped ? "warn" : "neutral", true);
    const segments = document.createElement("div");
    segments.className = "segments";
    for (let i = 0; i < max; i += 1) {
        const seg = document.createElement("div");
        const filled = i < count;
        seg.className = filled
            ? capped
                ? "segment segment--cap"
                : "segment segment--on"
            : "segment";
        segments.appendChild(seg);
    }
    el.appendChild(segments);
    return el;
}
function offerCard(offer) {
    const box = document.createElement("div");
    box.className = "obs__offer";
    const name = document.createElement("div");
    name.className = "obs__offer-name";
    name.textContent = offer.offer_name ?? offer.offer_id;
    box.appendChild(name);
    const id = document.createElement("code");
    id.className = "obs__offer-id";
    id.textContent = offer.offer_id;
    box.appendChild(id);
    if (typeof offer.apr_range === "string" && offer.apr_range) {
        const apr = document.createElement("div");
        apr.className = "obs__offer-apr";
        apr.textContent = `APR ${offer.apr_range}`;
        box.appendChild(apr);
    }
    if (typeof offer.rewards_summary === "string" && offer.rewards_summary) {
        const rewards = document.createElement("div");
        rewards.className = "obs__offer-meta";
        rewards.textContent = offer.rewards_summary;
        box.appendChild(rewards);
    }
    return box;
}
function collapsible(title, payload) {
    const details = document.createElement("details");
    details.className = "obs__details";
    const summary = document.createElement("summary");
    summary.textContent = title;
    details.appendChild(summary);
    const pre = document.createElement("pre");
    pre.className = "obs__json";
    pre.textContent = JSON.stringify(payload, null, 2);
    details.appendChild(pre);
    return details;
}
export function renderObservability(container) {
    container.replaceChildren();
    const result = state.lastResult;
    // --- Risk (hero) --------------------------------------------------
    container.appendChild(section("Customer risk"));
    container.appendChild(riskHero(result ?? { high_risk_probability: null }));
    // --- Turn KPIs ----------------------------------------------------
    container.appendChild(section("Latest turn"));
    if (!result) {
        container.appendChild(hint("Send a message to populate graph telemetry."));
    }
    else {
        const tiles = document.createElement("div");
        tiles.className = "obs__tiles";
        const stage = result.conversation_stage;
        tiles.appendChild(tile("Stage", stage ? (STAGE_LABELS[stage] ?? stage) : "n/a"));
        tiles.appendChild(result.risk_blocked
            ? tile("Guardrail", "Declined", "critical")
            : tile("Guardrail", "Passed", "good"));
        tiles.appendChild(clarifyTile(result));
        container.appendChild(tiles);
        if (result.selected_offer) {
            container.appendChild(section("Selected offer"));
            container.appendChild(offerCard(result.selected_offer));
        }
        container.appendChild(section("Graph state"));
        container.appendChild(collapsible("Accumulated features", result.accumulated_features));
        container.appendChild(collapsible("Candidate offers", (result.candidate_offers ?? []).map((offer) => ({
            offer_id: offer.offer_id,
            offer_name: offer.offer_name,
        }))));
    }
    // --- Session ------------------------------------------------------
    container.appendChild(section("Session"));
    const thread = document.createElement("div");
    thread.className = "obs__thread";
    thread.textContent = state.threadId;
    thread.title = state.threadId;
    container.appendChild(thread);
    const url = langsmithThreadUrl();
    if (url) {
        const link = document.createElement("a");
        link.className = "obs__link";
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "Open thread in LangSmith ↗";
        container.appendChild(link);
    }
    else {
        container.appendChild(hint("Set LANGSMITH_ORG to enable one-click thread links."));
    }
}
//# sourceMappingURL=observability.js.map