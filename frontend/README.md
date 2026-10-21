# Verity frontend

The interface for the document Q&A backend in `../backend`. React 18, Vite,
TypeScript strict, Tailwind. The design concept is documented in
`../docs/verity-how-it-works.html`; the tokens live in `src/index.css`.

## Run it

```sh
npm install
npm run dev          # http://127.0.0.1:5173, proxies /api to localhost:8000
npm run typecheck
npm run build
```

The dev server proxies `/api` to `VITE_PROXY_TARGET` (default
`http://localhost:8000`; compose sets `http://backend:8000`). The browser never
calls the model provider itself — the backend proxy is what keeps the API key
off the client, so there is no API base and no CORS in the normal path.

## Where things are

```
src/
  types.ts               API response shapes; the contract with the backend
  api/                   one fetch wrapper, one module per backend resource
  hooks/                 React Query hooks; server state and the session id
  lib/                   pure functions (citations, formatting, class names)
  components/
    AppShell.tsx         navigation frame, shared by every screen
    conversation/        the question composer, answer sheet, evidence, trace
    documents/           upload, status, corpus table
    ui/                  the few primitives shared across screens
```

`src/components/conversation/TurnView.tsx` renders one exchange.
`ConversationScreen.tsx` owns selection and layout: desktop puts evidence and
the run trace in a right-hand rail; below `lg` the same evidence list collapses
under the answer that cites it.

## The one rule worth knowing

The backend numbers the chunks it shows the model 1..N and the model cites
those numbers. `ordinal` in a citation is therefore that **shown-list
position**, not the order the citations happened to be appended — `[2]` in the
answer always points at evidence row 2.

History loaded from `GET /messages` has no persisted ordinal, so
`lib/citations.ts` detects the missing mapping and renders the markers as plain
text rather than inventing numbers. Same for the run trace: it is not stored, so
the panel says so instead of reconstructing a plausible-looking one.

## Design notes

- One accent (ochre) is reserved for pointing at evidence: citation numerals and
  the wordmark. Everything else is ink on archive paper.
- The answer is the only card-like object on screen; it is the artifact the
  reader came to inspect.
- Hairlines and spacing carry hierarchy; there are no gradient washes or
  decorative shadows.
- Body text is IBM Plex Serif at a reading-column width; identifiers, scores and
  locations are IBM Plex Mono, because they are data, not prose.
