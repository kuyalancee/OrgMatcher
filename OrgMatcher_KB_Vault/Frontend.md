# OrgMatcher — Frontend Specification

> Pass this document to Claude or Gemini CLI as the sole source of truth for building the frontend.

---

## Project Overview

**OrgMatcher** is an NLP-powered web app that matches UNT students to campus organizations. The user types a free-form description of what they want in a club (interests, hobbies, social vibe, goals), submits it, and receives the **top 5 matching organizations** ranked by cosine-similarity NLP scoring.

The frontend is a **React + Vite** single-page app that communicates with a **FastAPI** backend.

---

## Tech Stack

|Layer|Technology|
|---|---|
|Framework|React 18 + Vite|
|Styling|CSS Modules **or** plain CSS with CSS variables (no Tailwind)|
|HTTP|`fetch` API (no Axios)|
|Fonts|Google Fonts — load via `<link>` in `index.html`|
|Animations|CSS transitions/keyframes only — no animation libraries|
|Icons|None required; use Unicode/emoji sparingly if needed|

---

## Aesthetic Direction

**Tone:** Clean editorial — modern university branding meets a well-designed magazine.

**Key rules (non-negotiable):**

- **White (`#ffffff`) background everywhere.** No gradients on any background surface, ever.
- One strong accent color: `#00853E` (UNT's official green). Use it for interactive elements, highlights, and the logo mark.
- Secondary neutral: `#1a1a1a` for primary text. Use `#555` for secondary/caption text.
- Cards use a `1px solid #e5e5e5` border and a subtle `box-shadow: 0 2px 12px rgba(0,0,0,0.06)`. No background color on cards — they are white.
- **No gradients anywhere** — this is the single most important visual constraint.

**Typography:**

- Display / heading font: **"DM Serif Display"** (Google Fonts) — used for the app title and section labels.
- Body / UI font: **"DM Sans"** (Google Fonts) — used for all other text, inputs, buttons.
- Load both via a single `<link>` in `index.html`.

**Spacing system:** Use multiples of 8px (`8, 16, 24, 32, 48, 64`).

**Animations:**

- Page load: stagger-reveal the hero text and input with `opacity 0→1` + `translateY(12px→0)`, 300 ms ease, 80 ms delay between elements.
- Result cards: each card fades + slides in sequentially (`animation-delay: 0ms, 80ms, 160ms, 240ms, 320ms`).
- Button: `transform: scale(0.97)` on active press.
- Card hover: `transform: translateY(-3px)` + deepen box-shadow.
- All transitions: `200 ms ease` unless specified otherwise.

---

## File & Folder Structure

Create only these files inside `frontend/src/`:

```
frontend/
├── index.html                  ← Add Google Fonts <link> here
├── vite.config.js              ← Proxy /api → http://localhost:8000
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── App.css                 ← Global reset + CSS variables
    ├── components/
    │   ├── SearchBar.jsx       ← Textarea + submit button
    │   ├── SearchBar.css
    │   ├── ResultsGrid.jsx     ← Renders list of OrgCard
    │   ├── ResultsGrid.css
    │   ├── OrgCard.jsx         ← Single organization result card
    │   └── OrgCard.css
    └── assets/
        └── unt-logo.svg        ← (optional placeholder)
```

Do **not** create any other files. Do **not** add a router — this is a single view.

---

## CSS Variables (define in `App.css` `:root`)

```css
:root {
  --color-accent: #00853E;
  --color-accent-dark: #006830;
  --color-text-primary: #1a1a1a;
  --color-text-secondary: #555555;
  --color-border: #e5e5e5;
  --color-bg: #ffffff;
  --color-card-shadow: rgba(0, 0, 0, 0.06);

  --font-display: 'DM Serif Display', serif;
  --font-body: 'DM Sans', sans-serif;

  --radius-card: 12px;
  --radius-input: 8px;
  --radius-button: 8px;

  --shadow-card: 0 2px 12px var(--color-card-shadow);
  --shadow-card-hover: 0 8px 28px rgba(0, 0, 0, 0.11);

  --transition-fast: 200ms ease;
  --transition-medium: 300ms ease;
}
```

---

## Layout — `App.jsx`

Render a single-column, centered layout with max-width `720px` for the input section and `1100px` for the results grid.

```
┌─────────────────────────────────────────────────────┐
│  [Header]  OrgMatcher  (DM Serif Display, 2.4rem)   │
│  Tagline: "Find your people at UNT." (DM Sans)      │
├─────────────────────────────────────────────────────┤
│  [SearchBar]                                        │
│  ┌───────────────────────────────────────────────┐  │
│  │  Textarea (4 rows)                            │  │
│  │  placeholder: "I enjoy..."                   │  │
│  └───────────────────────────────────────────────┘  │
│  [ Find My Orgs → ]  (accent green button)          │
├─────────────────────────────────────────────────────┤
│  [ResultsGrid]  — shown only after a response       │
│  "Your Top Matches" (section heading)               │
│  ┌──────┐ ┌──────┐ ┌──────┐                        │
│  │Card 1│ │Card 2│ │Card 3│                        │
│  └──────┘ └──────┘ └──────┘                        │
│  ┌──────┐ ┌──────┐                                 │
│  │Card 4│ │Card 5│                                 │
│  └──────┘ └──────┘                                 │
└─────────────────────────────────────────────────────┘
```

- Header is centered, `padding-top: 64px`.
- The green accent dot or underscore on "OrgMatcher" title is optional but encouraged.
- No navbar, no footer.

---

## Component Specifications

### `SearchBar.jsx`

**Props:** `onSubmit(query: string)`, `isLoading: boolean`

**Behavior:**

- Renders a `<textarea>` (4 rows) and a `<button>`.
- On submit: trim whitespace; if empty, shake the textarea (CSS keyframe, 300 ms).
- Disable the button and show `"Matching…"` text while `isLoading` is true.
- Button is full-width below the textarea on mobile, right-aligned on desktop.

**Textarea styles:**

- `width: 100%`, `padding: 14px 16px`, `font-size: 1rem`, `font-family: var(--font-body)`
- `border: 1.5px solid var(--color-border)`, `border-radius: var(--radius-input)`
- On focus: `border-color: var(--color-accent)`, `outline: none`, `box-shadow: 0 0 0 3px rgba(0,133,62,0.12)`
- Resize: `vertical` only.

**Button styles:**

- Background: `var(--color-accent)`, color: `#fff`, `font-family: var(--font-body)`, `font-size: 1rem`, `font-weight: 600`
- `padding: 12px 28px`, `border: none`, `border-radius: var(--radius-button)`, `cursor: pointer`
- Hover: `background: var(--color-accent-dark)`
- Active: `transform: scale(0.97)`
- Disabled (loading): `opacity: 0.6`, `cursor: not-allowed`

---

### `ResultsGrid.jsx`

**Props:** `results: OrgResult[]`

**Behavior:**

- Hidden when `results` is empty/null.
- Section heading: `"Your Top Matches"` in `var(--font-display)`, `1.6rem`, left-aligned.
- Renders a CSS grid: `grid-template-columns: repeat(auto-fill, minmax(300px, 1fr))`, `gap: 24px`.
- Each child `OrgCard` gets `animation-delay: N * 80ms` (N = 0–4).

---

### `OrgCard.jsx`

**Props:**

```typescript
{
  name: string,
  acronym: string,
  summary: string,
  description: string,
  image_url: string | null,
  org_url: string,
  rank: number          // 1–5, for the rank badge
}
```

**Card anatomy (top to bottom):**

```
┌──────────────────────────────────────┐
│ [Image]  aspect-ratio 16/9, cover    │  ← gray placeholder if null
│          border-radius top corners   │
├──────────────────────────────────────┤
│ [Rank badge]  "#1"  (accent green)   │  ← absolute, top-right of image
├──────────────────────────────────────┤
│ [Name]   DM Serif Display, 1.2rem    │
│ [Acronym]  DM Sans, 0.85rem, muted   │
│ [Summary]  DM Sans, 0.9rem, 3 lines  │  ← clamp to 3 lines with ellipsis
│            overflow-hidden           │
│                                      │
│ [ Visit Organization → ]             │  ← text link, accent green
└──────────────────────────────────────┘
```

**Card styles:**

- `background: var(--color-bg)`, `border: 1px solid var(--color-border)`
- `border-radius: var(--radius-card)`, `box-shadow: var(--shadow-card)`
- `overflow: hidden`, `display: flex; flex-direction: column`
- Hover: `transform: translateY(-3px)`, `box-shadow: var(--shadow-card-hover)`, `transition: var(--transition-fast)`

**Image:**

- `<img>` with `width: 100%`, `aspect-ratio: 16/9`, `object-fit: cover`
- If `image_url` is null/empty: show a `div` placeholder with `background: #f0f0f0` and a centered `🏛️` emoji at `2rem`.

**Rank badge:**

- `position: absolute; top: 10px; right: 10px`
- `background: var(--color-accent)`, `color: #fff`, `font-family: var(--font-body)`, `font-weight: 700`, `font-size: 0.75rem`
- `padding: 3px 8px`, `border-radius: 999px`

**"Visit Organization" link:**

- `<a href={org_url} target="_blank" rel="noopener noreferrer">`
- Color: `var(--color-accent)`, `font-weight: 600`, `font-size: 0.9rem`
- No underline by default; underline on hover.
- Margin-top: `auto` (pushes to bottom of card).

---

## API Integration

### Vite proxy (`vite.config.js`)

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
```

### API call (inside `App.jsx`)

**Endpoint:** `POST /api/match`

**Request body:**

```json
{ "query": "<user input string>" }
```

**Expected response:**

```json
{
  "results": [
    {
      "name": "string",
      "acronym": "string",
      "summary": "string",
      "description": "string",
      "image_url": "string | null",
      "org_url": "string",
      "rank": 1
    }
  ]
}
```

**State model in `App.jsx`:**

```js
const [query, setQuery] = useState('')
const [results, setResults] = useState([])
const [isLoading, setIsLoading] = useState(false)
const [error, setError] = useState(null)
```

**Fetch logic:**

```js
async function handleSearch(query) {
  setIsLoading(true)
  setError(null)
  try {
    const res = await fetch('/api/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    })
    if (!res.ok) throw new Error(`Server error: ${res.status}`)
    const data = await res.json()
    setResults(data.results)
  } catch (err) {
    setError('Something went wrong. Please try again.')
  } finally {
    setIsLoading(false)
  }
}
```

**Error display:** A simple centered `<p>` in `color: #c0392b`, `font-size: 0.9rem`, shown below the SearchBar when `error` is not null.

---

## Loading State

While `isLoading` is true and `results` is empty, show a **skeleton grid** in place of the ResultsGrid:

- 5 skeleton cards in the same grid layout.
- Each skeleton card is the same dimensions as a real card.
- Skeleton uses a `@keyframes shimmer` animation: `background` sweeps from `#f0f0f0` to `#e0e0e0` to `#f0f0f0` over `1.4s` using `background-size: 200%` and `background-position` animation.
- No spinner, no loading text in the grid area (the button already says "Matching…").

---

## Responsive Breakpoints

|Breakpoint|Behavior|
|---|---|
|`> 1024px`|3-column results grid|
|`640px – 1024px`|2-column results grid|
|`< 640px`|1-column results grid; button full-width|

Use `@media` queries inside each component's CSS file. Do not use a global breakpoint file.

---

## Accessibility

- `<textarea>` must have `aria-label="Describe what you're looking for in an organization"`
- `<button>` text must update to `"Matching…"` when loading (screen-reader friendly).
- All `<img>` elements must have `alt={org.name}`.
- Skeleton cards must have `aria-hidden="true"`.
- Color contrast: all text on white background must meet WCAG AA (accent green `#00853E` on white is AA-compliant for large text; verify small text uses `#1a1a1a`).

---

## What NOT to Do

- ❌ No Tailwind, no CSS-in-JS, no styled-components
- ❌ No gradients on any background, ever
- ❌ No navbar, sidebar, footer, or multi-page routing
- ❌ No external animation libraries (Framer Motion, GSAP, etc.)
- ❌ No `Inter`, `Roboto`, or `Arial` fonts
- ❌ No purple, blue, or dark-mode theme — white background only
- ❌ Do not hardcode any org data — all data comes from the API
- ❌ Do not add TypeScript — stay in plain JSX

---

## Deliverables Checklist

The implementation is complete when:

- [ ] `index.html` loads DM Serif Display + DM Sans from Google Fonts
- [ ] `vite.config.js` proxies `/api` to `localhost:8000`
- [ ] `App.jsx` manages all state and renders header + SearchBar + ResultsGrid
- [ ] `SearchBar.jsx` submits query, shows loading state, shakes on empty submit
- [ ] `ResultsGrid.jsx` renders 5 `OrgCard` components with staggered animation
- [ ] `OrgCard.jsx` displays image (with placeholder), rank badge, name, acronym, summary, and visit link
- [ ] Skeleton loading state shows 5 shimmer cards while fetching
- [ ] Error state displays a message below the search bar
- [ ] All responsive breakpoints are handled
- [ ] No gradients exist anywhere in the codebase
- [ ] All accessibility requirements are met