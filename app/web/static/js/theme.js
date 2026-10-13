/**
 * Theme selection.
 *
 * Light is the default, always - not "whatever the OS prefers". Dark is a
 * deliberate choice the viewer makes and we remember. That is why there is no
 * `prefers-color-scheme` query anywhere in the stylesheet: the OS setting is
 * intentionally not consulted.
 *
 * The applied value lives as `data-theme` on <html>, which is what the token
 * blocks in styles.css key off. `index.html` also sets it inline in <head>
 * before first paint; without that the page would paint light and then flip,
 * which is very visible on reload.
 */
/** Shared with the inline bootstrap script in index.html - keep them in sync. */
export const STORAGE_KEY = "nba-theme";
const DEFAULT_THEME = "light";
/**
 * Read the stored preference.
 *
 * Every localStorage access is guarded: it throws outright in a private window
 * and when site data is blocked. A theme preference must never be able to take
 * the app down, so failure just means "use the default".
 */
export function storedTheme() {
    try {
        const value = localStorage.getItem(STORAGE_KEY);
        return value === "light" || value === "dark" ? value : null;
    }
    catch {
        return null;
    }
}
export function currentTheme() {
    const attr = document.documentElement.getAttribute("data-theme");
    if (attr === "light" || attr === "dark") {
        return attr;
    }
    return storedTheme() ?? DEFAULT_THEME;
}
export function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
        localStorage.setItem(STORAGE_KEY, theme);
    }
    catch {
        // Non-fatal: the theme still applies for this page view, it just will not
        // survive a reload.
    }
}
export function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    return next;
}
/**
 * Point the toggle button at the theme it will switch *to*, not the one in
 * use - that is what the label and the icon should describe.
 */
export function syncToggleButton(button, theme) {
    const goingTo = theme === "dark" ? "light" : "dark";
    const label = `Switch to ${goingTo} theme`;
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-pressed", String(theme === "dark"));
    button.title = label;
    // Sun when dark is active (click for light), moon when light is active.
    button.textContent = theme === "dark" ? "\u2600" : "\u263D";
}
//# sourceMappingURL=theme.js.map