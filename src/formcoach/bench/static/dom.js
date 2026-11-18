// Small DOM helpers for the Bench pages. Server text always goes in as text, never as HTML.

/**
 * Make an element: `h("button", {class: "btn", onclick: fn}, "Scan")`. Attributes starting with
 * "on" become listeners; `hidden`/`disabled`/`checked`/`value`/`selected` are set as properties;
 * children are nodes or text (null/false skipped).
 */
export function h(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs ?? {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "class") e.className = v;
    else if (["hidden", "disabled", "checked", "value", "selected"].includes(k)) e[k] = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    e.append(c instanceof Node ? c : String(c));
  }
  return e;
}

/** Set `el.textContent` only when it differs (no layout work for unchanged text). */
export function setText(el, text) {
  const s = String(text ?? "");
  if (el.textContent !== s) el.textContent = s;
}

/** Local wall-clock time "HH:MM:SS" of `ms` (ms since epoch). */
export function clockOf(ms) {
  const d = new Date(ms);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
