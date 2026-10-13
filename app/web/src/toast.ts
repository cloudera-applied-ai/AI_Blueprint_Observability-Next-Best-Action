/** Transient corner notifications — the replacement for `st.toast`. */

const TOAST_MS = 3200;

export function showToast(message: string, kind: "info" | "error" = "info"): void {
  const host = document.getElementById("toasts");
  if (!host) {
    return;
  }

  const toast = document.createElement("div");
  toast.className = `toast toast--${kind}`;
  toast.setAttribute("role", kind === "error" ? "alert" : "status");
  toast.textContent = message;
  host.appendChild(toast);

  window.setTimeout(() => {
    toast.classList.add("toast--leaving");
    // Matches the .toast--leaving transition; removing sooner would cut the
    // fade off mid-way.
    window.setTimeout(() => {
      toast.remove();
    }, 250);
  }, TOAST_MS);
}
