# Backend-Frontend Integration — 2026-04-10

## Goal / Issue Overview
`main.py` was a stub (only `# FIXME` and one import). The backend needed a working FastAPI app exposing the endpoint the frontend calls: `POST /api/match`. The `tfidf.py` `search()` return shape also didn't match the frontend contract.

## Key Decisions

- **Endpoint is `/api/match`** (not `/match`): The Vite proxy forwards `/api/*` to `localhost:8000` without rewriting the path, so FastAPI must handle the full `/api/match` path.
- **Lifespan for engine init**: `OrgMatcher` is initialized once at startup via FastAPI's `asynccontextmanager` lifespan, not at import time. This is the recommended pattern for FastAPI 0.93+.
- **`short_name` renamed to `acronym` in `search()`**: The frontend contract uses `acronym`; the DB column is `short_name`. Renaming happens in `tfidf.py` so `main.py` stays clean.
- **`image_url` / `org_url` are optional**: If the DB has those columns they're returned; if not, `main.py` gracefully falls back to `None`/`""` via `.get()`. No changes to the DB schema needed.
- **`rank` assigned in `main.py`**: It's a presentation concern, not a search concern, so `search()` doesn't produce it — `main.py` sets it as `i + 1` over the ordered results.

## Files Modified
- `backend/main.py` — Full rewrite: FastAPI app, CORS, Pydantic models, lifespan init, `POST /api/match` endpoint
- `backend/tfidf.py` — `search()` updated to rename `short_name` → `acronym` and include `image_url`/`org_url` if present in the DB

To run the frontend,
```
cd frontend/OrgMatcher-app; npm run dev
```
To run the backend,
```
cd backend; uvicorn main:app --reload
```