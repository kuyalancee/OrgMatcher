# CORS Troubleshooting: Trailing Slash Issue

## The Error
When the Vercel frontend (`https://org-matcher.vercel.app`) attempted to communicate with the Google Cloud Run backend (`https://orgmatcher-353106949537.us-south1.run.app/api/match`), the browser blocked the request with the following error:

```
Access to fetch at 'https://orgmatcher-353106949537.us-south1.run.app/api/match' from origin 'https://org-matcher.vercel.app' has been blocked by CORS policy: Response to preflight request doesn't pass access control check: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

## The Cause
The CORS (Cross-Origin Resource Sharing) middleware in FastAPI requires an **exact string match** for the allowed origins. 

In `backend/main.py`, the default frontend URL was defined with a trailing slash:
```python
frontend_url = os.environ.get("FRONTEND_URL", "https://org-matcher.vercel.app/") # Incorrect
```

However, the browser sends the `Origin` header without the trailing slash: `https://org-matcher.vercel.app`. Because `https://org-matcher.vercel.app/` does not exactly match `https://org-matcher.vercel.app`, FastAPI rejected the request.

## The Fix
To resolve this, the trailing slash must be removed from the allowed origins list in the backend configuration.

**In `backend/main.py`:**
```python
frontend_url = os.environ.get("FRONTEND_URL", "https://org-matcher.vercel.app") # Correct (No trailing slash)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        frontend_url,
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Additionally, ensure that if the `FRONTEND_URL` environment variable is explicitly set in Google Cloud Run, it also does not contain a trailing slash. After updating the code or environment variable, the backend service must be redeployed.
