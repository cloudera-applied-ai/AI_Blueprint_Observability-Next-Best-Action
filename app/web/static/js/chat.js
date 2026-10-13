/**
 * Chat transcript rendering: message bubbles, feedback, copy, typing state.
 */
import { postFeedback } from "./api.js";
import { state } from "./state.js";
import { showToast } from "./toast.js";
/** Opening lines offered on the empty state; clicking one sends it. */
export const SUGGESTIONS = [
    "Hi, I'd like to upgrade my credit card.",
    "I'm looking for something with cashback on everyday spending.",
    "Can you help me consolidate a balance from another card?",
];
/**
 * Render assistant text as HTML.
 *
 * The model emits light markdown (bold, bullets, paragraphs). Rather than
 * pull in a markdown dependency for four constructs, escape everything first
 * and then re-introduce exactly the tags we want. Escaping first is the point:
 * LLM output is untrusted text, and a customer could get it to echo markup.
 */
export function renderMarkdown(text) {
    const escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    const lines = escaped.split("\n");
    const out = [];
    let inList = false;
    for (const line of lines) {
        const trimmed = line.trim();
        const bullet = /^[-*]\s+(.*)$/.exec(trimmed);
        if (bullet) {
            if (!inList) {
                out.push("<ul>");
                inList = true;
            }
            out.push(`<li>${inline(bullet[1] ?? "")}</li>`);
            continue;
        }
        if (inList) {
            out.push("</ul>");
            inList = false;
        }
        if (trimmed) {
            out.push(`<p>${inline(trimmed)}</p>`);
        }
    }
    if (inList) {
        out.push("</ul>");
    }
    return out.join("");
}
/** Bold and inline code, applied to already-escaped text. */
function inline(text) {
    return text
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>");
}
function formatTime(at) {
    return at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function feedbackButton(label, title, score, runId) {
    const button = document.createElement("button");
    button.className = "thumb";
    button.type = "button";
    button.textContent = label;
    button.title = title;
    button.setAttribute("aria-label", title);
    button.addEventListener("click", () => {
        // Lock both thumbs for this turn immediately. Re-clicking would write a
        // second feedback record to the same run, which skews the feedback rate
        // the LangSmith dashboards report.
        const group = button.parentElement;
        group?.querySelectorAll("button.thumb").forEach((b) => {
            b.setAttribute("disabled", "true");
        });
        button.classList.add("thumb--chosen");
        void postFeedback(runId, score)
            .then((result) => {
            if (result.ok) {
                showToast("Thanks for the feedback!");
            }
            else {
                showToast(`LangSmith feedback failed: ${result.error ?? "unknown"}`, "error");
            }
        })
            .catch((err) => {
            showToast(`LangSmith feedback failed: ${err instanceof Error ? err.message : String(err)}`, "error");
        });
    });
    return button;
}
function copyButton(text) {
    const button = document.createElement("button");
    button.className = "msg__copy";
    button.type = "button";
    button.textContent = "Copy";
    button.title = "Copy this reply";
    button.setAttribute("aria-label", "Copy this reply");
    button.addEventListener("click", () => {
        // navigator.clipboard needs a secure context and can be denied outright,
        // so never assume it resolves.
        void navigator.clipboard
            ?.writeText(text)
            .then(() => {
            button.textContent = "Copied";
            window.setTimeout(() => {
                button.textContent = "Copy";
            }, 1600);
        })
            .catch(() => {
            showToast("Could not copy to the clipboard.", "error");
        });
    });
    return button;
}
function messageElement(message) {
    const wrapper = document.createElement("div");
    wrapper.className = `msg msg--${message.role}`;
    if (message.isError) {
        wrapper.classList.add("msg--error");
        const bubble = document.createElement("div");
        bubble.className = "msg__bubble";
        const icon = document.createElement("span");
        icon.className = "msg__error-icon";
        icon.textContent = "!";
        icon.setAttribute("aria-hidden", "true");
        const body = document.createElement("span");
        body.textContent = message.content;
        bubble.append(icon, body);
        wrapper.appendChild(bubble);
        return wrapper;
    }
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    if (message.role === "assistant") {
        bubble.innerHTML = renderMarkdown(message.content);
    }
    else {
        // User text is never markdown-rendered; textContent keeps it inert.
        bubble.textContent = message.content;
    }
    wrapper.appendChild(bubble);
    const actions = document.createElement("div");
    actions.className = "msg__actions";
    if (message.role === "assistant") {
        if (message.runId) {
            const runId = message.runId;
            actions.appendChild(feedbackButton("\u{1F44D}", "Helpful", 1, runId));
            actions.appendChild(feedbackButton("\u{1F44E}", "Not helpful", 0, runId));
        }
        actions.appendChild(copyButton(message.content));
        if (message.runId) {
            const trace = document.createElement("span");
            trace.className = "msg__runid";
            trace.textContent = `run ${message.runId.slice(0, 8)}`;
            trace.title = `LangSmith run_id: ${message.runId}`;
            actions.appendChild(trace);
        }
    }
    const time = document.createElement("span");
    time.className = "msg__time";
    time.textContent = formatTime(message.at);
    actions.appendChild(time);
    wrapper.appendChild(actions);
    return wrapper;
}
/** Bouncing-dots placeholder shown while a turn is in flight. */
function typingElement() {
    const wrapper = document.createElement("div");
    wrapper.className = "msg msg--assistant msg--typing";
    wrapper.innerHTML =
        '<div class="msg__bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
    return wrapper;
}
/**
 * Onboarding empty state.
 *
 * `onSuggestion` is wired by main.ts so a click sends the line straight
 * through - the fastest path from a cold open to a traced turn.
 */
function emptyState(onSuggestion) {
    const wrapper = document.createElement("div");
    wrapper.className = "chat__empty";
    const icon = document.createElement("div");
    icon.className = "chat__empty-icon";
    icon.textContent = "◆";
    icon.setAttribute("aria-hidden", "true");
    wrapper.appendChild(icon);
    const heading = document.createElement("h2");
    heading.textContent = "Start a conversation";
    wrapper.appendChild(heading);
    const blurb = document.createElement("p");
    blurb.textContent =
        "The agent asks clarifying questions, then recommends a card. LOW and MED risk customers reach an offer; HIGH risk trips the guardrail.";
    wrapper.appendChild(blurb);
    const list = document.createElement("div");
    list.className = "chat__suggestions";
    for (const text of SUGGESTIONS) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "suggestion";
        button.textContent = text;
        const arrow = document.createElement("span");
        arrow.className = "suggestion__arrow";
        arrow.textContent = "→";
        arrow.setAttribute("aria-hidden", "true");
        button.appendChild(arrow);
        button.addEventListener("click", () => {
            onSuggestion(text);
        });
        list.appendChild(button);
    }
    wrapper.appendChild(list);
    return wrapper;
}
/**
 * Repaint the whole transcript.
 *
 * Full re-render on every turn: the transcript is a handful of messages, so
 * this costs nothing measurable and removes a class of incremental-update
 * bugs entirely.
 */
export function renderChat(container, onSuggestion) {
    container.replaceChildren();
    if (state.history.length === 0 && !state.pending) {
        container.appendChild(emptyState(onSuggestion));
        return;
    }
    for (const message of state.history) {
        container.appendChild(messageElement(message));
    }
    if (state.pending) {
        container.appendChild(typingElement());
    }
    container.scrollTop = container.scrollHeight;
}
//# sourceMappingURL=chat.js.map