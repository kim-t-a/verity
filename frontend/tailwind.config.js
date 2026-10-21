/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Every colour reads a CSS variable, so the dark-mode block in
      // index.css restyles the whole app without a second set of utilities.
      // Do not use Tailwind's "/opacity" modifiers on these: a var() colour
      // has no alpha channel to rewrite.
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        fg: "var(--fg)",
        muted: "var(--fg-muted)",
        faint: "var(--fg-faint)",
        rule: "var(--rule)",
        "rule-strong": "var(--rule-strong)",
        accent: "var(--accent)",
        "accent-wash": "var(--accent-wash)",
        "accent-rule": "var(--accent-rule)",
        ok: "var(--ok)",
        "ok-wash": "var(--ok-wash)",
        stop: "var(--stop)",
        "stop-wash": "var(--stop-wash)",
      },
      fontFamily: {
        display: ["var(--font-display)"],
        body: ["var(--font-body)"],
        mono: ["var(--font-mono)"],
      },
      maxWidth: {
        col: "68ch",
        wide: "1060px",
      },
      boxShadow: {
        sheet: "var(--shadow)",
      },
      transitionDuration: {
        quick: "120ms",
      },
    },
  },
  plugins: [],
};
