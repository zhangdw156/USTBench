# Math markup in `index.html`

The deck loads **KaTeX** and **auto-render** on `#deck`. You can write formulas the same way as in many **Markdown** + math setups.

## Supported delimiters (paste into HTML)

| Style | Example | Note |
|--------|---------|------|
| Inline (LaTeX) | `\(E=\langle S,A,O,T\rangle\)` | Safe in running text |
| Inline (MD-style) | `$E=\langle S,A,O,T\rangle$` | Same as Typora / VS Code math |
| Display | `\[ \sum_{i=1}^n x_i \]` or `$$\sum_{i=1}^n x_i$$` | Centered block |

## Workflow from Markdown

1. In your `.md` notes, write: `$T(s,a)\mapsto s'$`.
2. Export or copy the **HTML** your tool produces (often it becomes `\(...\)`), **or** paste the `$...$` string directly into `index.html` — both work after refresh.

## Caveat

Avoid bare `$` for currency (e.g. `$5`) in the same paragraph as math, or escape / rephrase — single-`$` delimiters can mis-parse. Prefer `\(...\)` for money + math on one slide.
