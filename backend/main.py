from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tfidf import OrgMatcher

# models

class MatchRequest(BaseModel):
    query: str

class OrgResult(BaseModel):
    name: str
    acronym: str
    summary: str
    description: str
    image_url: str | None
    org_url: str
    rank: int

class MatchResponse(BaseModel):
    results: list[OrgResult]

# app; init engine at startup once

engine: OrgMatcher | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = OrgMatcher()
    yield
    engine = None

# fastapi

app = FastAPI(lifespan=lifespan)

import os

frontend_url = os.environ.get("FRONTEND_URL", "https://org-matcher.vercel.app/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        frontend_url,
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# http endpoints

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/api/match", response_model=MatchResponse)
def match_orgs(body: MatchRequest):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    raw = engine.search(query, top_n=5)
    results = [
        OrgResult(
            name=r["name"],
            acronym=r["acronym"],
            summary=r["summary"],
            description=r["description"],
            image_url=r.get("picture"),
            org_url=r.get("url") or "",
            rank=i + 1,
        )
        for i, r in enumerate(raw)
    ]
    return MatchResponse(results=results)
