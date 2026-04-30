# Deployment Strategy

## Overview
- **Frontend:** Vercel (Vite + React)
- **Backend:** Google Cloud Platform (GCP) Cloud Run (FastAPI + Python)

## Frontend (Vercel)
- **Pros:** Zero-configuration deployments for Vite, automatic CI/CD on GitHub push, excellent global edge caching.
- **Action Items:**
  - Update API calls to use an environment variable (e.g., `VITE_API_URL`) instead of hardcoding `http://localhost:8000`.
  - Set the `VITE_API_URL` environment variable in Vercel project settings to point to the GCP backend URL once deployed.

## Backend (GCP Cloud Run & Docker)
It is highly recommended (and practically required for Cloud Run) to containerize the backend using Docker.

- **Why Cloud Run:**
  - **Cost-effective:** Scales to zero when idle, meaning you only pay for active processing time.
  - **Read-only Database:** Since the backend only reads from `unt_clubs.db` and does not write to it, you don't need a managed database like Cloud SQL. The SQLite `.db` file can be copied directly into the Docker image.

### Key Suggestions & Action Items

**1. Update CORS Configuration**
The backend currently only allows requests from the local frontend. Update `backend/main.py` to include the future Vercel URL in the `allow_origins` list:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://your-app-name.vercel.app" # <-- Add this once deployed to Vercel
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**2. Pre-download NLTK Data in Docker**
The backend downloads NLTK `wordnet` data on startup if it's missing. To avoid slow "cold starts" in Cloud Run, pre-download this data during the Docker build process to bake it directly into the image:
```dockerfile
# Add to future Dockerfile
RUN python -m nltk.downloader wordnet
```

**3. Optimize "Cold Start" Times**
The application initializes `OrgMatcher` on startup, which reads the SQLite DB, lemmatizes text, and calculates the TF-IDF matrix. When Cloud Run spins up a new container from zero, the first user will experience a "cold start" delay.
- **If initialization is fast enough (< 2-3 seconds):** No changes needed.
- **If initialization is too slow:** Consider pre-computing the `tfidf_matrix` and saving it as a `.pkl` (Pickle) file locally. The server can then instantly load the pre-calculated matrix into memory instead of recalculating it on every boot.
