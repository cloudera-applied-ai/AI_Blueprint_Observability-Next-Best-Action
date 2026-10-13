/**
 * Entry point: boots config + the customer slate, then wires the composer,
 * the customer picker, the theme toggle and "New conversation".
 */

import { ApiError, fetchConfig, fetchCustomers, postTurn } from "./api.js";
import { renderChat } from "./chat.js";
import { renderObservability } from "./observability.js";
import { resetConversation, state } from "./state.js";
import { currentTheme, syncToggleButton, toggleTheme } from "./theme.js";
import { showToast } from "./toast.js";

interface Elements {
  chat: HTMLElement;
  observability: HTMLElement;
  form: HTMLFormElement;
  input: HTMLTextAreaElement;
  send: HTMLButtonElement;
  customerSelect: HTMLSelectElement;
  customerHint: HTMLElement;
  newConversation: HTMLButtonElement;
  themeToggle: HTMLButtonElement;
  scrollBottom: HTMLButtonElement;
}

function requireElements(): Elements {
  const byId = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) {
      throw new Error(`missing #${id} in index.html`);
    }
    return el as T;
  };

  return {
    chat: byId("chat"),
    observability: byId("observability"),
    form: byId<HTMLFormElement>("composer"),
    input: byId<HTMLTextAreaElement>("composer-input"),
    send: byId<HTMLButtonElement>("composer-send"),
    customerSelect: byId<HTMLSelectElement>("customer-select"),
    customerHint: byId("customer-hint"),
    newConversation: byId<HTMLButtonElement>("new-conversation"),
    themeToggle: byId<HTMLButtonElement>("theme-toggle"),
    scrollBottom: byId<HTMLButtonElement>("scroll-bottom"),
  };
}

function formatCustomerLabel(
  name: string,
  tier: string,
  income: number,
  id: number,
): string {
  const money = income.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
  return `#${id} — ${name} (${tier} risk, ${money})`;
}

/** Shimmer placeholder while `/api/customers` is in flight. */
function showCustomerSkeleton(el: Elements): void {
  el.customerSelect.style.display = "none";
  el.customerSelect.insertAdjacentHTML(
    "afterend",
    '<div id="customer-skeleton" class="skeleton skeleton--select"></div>',
  );
  el.input.disabled = true;
  el.send.disabled = true;
}

function clearCustomerSkeleton(el: Elements): void {
  document.getElementById("customer-skeleton")?.remove();
  el.customerSelect.style.display = "";
}

function populateCustomers(el: Elements): void {
  clearCustomerSkeleton(el);
  el.customerSelect.replaceChildren();

  if (state.customers.length === 0) {
    const option = document.createElement("option");
    option.textContent = "No customers found";
    el.customerSelect.appendChild(option);
    el.customerSelect.disabled = true;
    el.customerHint.textContent =
      "Database not seeded. Run: python app/pii_datagen.py --ensure";
    el.customerHint.classList.add("hint--warn");
    // No customer means no risk tier, no eligibility filter and no offer - so
    // keep the composer disabled rather than failing one turn at a time.
    el.input.disabled = true;
    el.send.disabled = true;
    return;
  }

  el.customerSelect.disabled = false;
  el.input.disabled = false;

  for (const customer of state.customers) {
    const option = document.createElement("option");
    option.value = String(customer.customer_id);
    option.textContent = formatCustomerLabel(
      customer.full_name,
      customer.risk_tier,
      customer.annual_income,
      customer.customer_id,
    );
    el.customerSelect.appendChild(option);
  }

  const first = state.customers[0];
  if (first) {
    state.customerId = first.customer_id;
    el.customerSelect.value = String(first.customer_id);
  }
  updateCustomerHint(el);
}

/** Say up front which path the graph will take for the selected tier. */
function updateCustomerHint(el: Elements): void {
  const customer = state.customers.find(
    (c) => c.customer_id === state.customerId,
  );
  if (!customer) {
    return;
  }

  el.customerHint.classList.remove("hint--warn");
  el.customerHint.textContent =
    customer.risk_tier === "HIGH"
      ? "HIGH risk → the guardrail declines and redirects to support."
      : `${customer.risk_tier} risk → clarify, then present an offer.`;
}

function repaint(el: Elements): void {
  renderChat(el.chat, (text) => {
    void submitTurn(el, text);
  });
  renderObservability(el.observability);
  el.send.disabled = state.pending || el.input.disabled;
  updateScrollButton(el);
}

/** Only offer "scroll to latest" when the latest is actually off-screen. */
function updateScrollButton(el: Elements): void {
  const distance =
    el.chat.scrollHeight - el.chat.scrollTop - el.chat.clientHeight;
  el.scrollBottom.classList.toggle("scroll-bottom--visible", distance > 120);
}

async function submitTurn(el: Elements, message: string): Promise<void> {
  if (state.pending || el.input.disabled) {
    return;
  }

  state.history.push({
    role: "user",
    content: message,
    at: new Date(),
    runId: null,
  });
  state.pending = true;
  repaint(el);

  try {
    const result = await postTurn(message, state.threadId, state.customerId);
    state.history.push({
      role: "assistant",
      content: result.display_message || result.assistant_message,
      at: new Date(),
      runId: result.run_id,
    });
    state.lastResult = result;
  } catch (err: unknown) {
    const detail =
      err instanceof ApiError
        ? `${err.message} (HTTP ${err.status})`
        : err instanceof Error
          ? err.message
          : String(err);
    // Surfaced in the transcript rather than only as a toast, so the failed
    // turn stays visible in context when you scroll back.
    state.history.push({
      role: "assistant",
      content: `Something went wrong on that turn: ${detail}`,
      at: new Date(),
      runId: null,
      isError: true,
    });
    showToast("Turn failed — see the transcript for details.", "error");
  } finally {
    state.pending = false;
    repaint(el);
  }
}

function wire(el: Elements): void {
  el.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = el.input.value.trim();
    if (!message) {
      return;
    }
    el.input.value = "";
    resizeInput(el.input);
    void submitTurn(el, message);
  });

  // Enter sends, Shift+Enter makes a newline — the convention people expect
  // from a chat box, which a bare <textarea> does not give you.
  el.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.form.requestSubmit();
    }
  });

  el.input.addEventListener("input", () => {
    resizeInput(el.input);
  });

  el.chat.addEventListener("scroll", () => {
    updateScrollButton(el);
  });

  el.scrollBottom.addEventListener("click", () => {
    el.chat.scrollTop = el.chat.scrollHeight;
  });

  el.customerSelect.addEventListener("change", () => {
    state.customerId = Number(el.customerSelect.value);
    updateCustomerHint(el);
    // Switching customer mid-thread would leave the checkpointer holding the
    // previous customer's record, so start a clean thread.
    resetConversation();
    repaint(el);
  });

  el.newConversation.addEventListener("click", () => {
    resetConversation();
    repaint(el);
    showToast("Started a new conversation.");
    el.input.focus();
  });

  el.themeToggle.addEventListener("click", () => {
    syncToggleButton(el.themeToggle, toggleTheme());
  });
}

/** Grow the composer with its content, up to a cap. */
function resizeInput(input: HTMLTextAreaElement): void {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

async function boot(): Promise<void> {
  const el = requireElements();
  syncToggleButton(el.themeToggle, currentTheme());
  wire(el);
  showCustomerSkeleton(el);
  repaint(el);

  try {
    const [config, customers] = await Promise.all([
      fetchConfig(),
      fetchCustomers(),
    ]);
    state.config = config;
    state.customers = customers;
  } catch (err: unknown) {
    showToast(
      `Could not reach the API: ${err instanceof Error ? err.message : String(err)}`,
      "error",
    );
  }

  populateCustomers(el);
  repaint(el);
  el.input.focus();
}

void boot();
