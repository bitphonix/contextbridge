from ddtrace import patch_all
patch_all()

import sentry_sdk
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from app.graph.pipeline import pipeline
from app.db.mongo import get_brain, get_all_brains

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=1.0,
    environment=os.getenv("APP_ENV", "development"),
)

app = FastAPI(
    title="ContextBridge",
    description="Extract portable context brains from AI conversations",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


class ExtractRequest(BaseModel):
    conversation: str
    user_id: Optional[str] = None


class ExtractResponse(BaseModel):
    brain_id:      str
    domain:        str
    goal:          str
    decisions:     list
    dead_ends:     list
    current_state: str
    next_steps:    list
    briefing:      str
    critic_score:  int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    if len(req.conversation.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Conversation too short. Paste at least a few messages."
        )

    initial_state = {
        "raw_conversation": req.conversation,
        "user_id":          req.user_id,
        "retry_count":      0,
    }

    result = pipeline.invoke(initial_state)

    if result.get("error") and not result.get("brain_id"):
        raise HTTPException(status_code=500, detail=result["error"])

    return ExtractResponse(
        brain_id=      result.get("brain_id", ""),
        domain=        result.get("domain", "general"),
        goal=          result.get("goal", ""),
        decisions=     result.get("decisions", []),
        dead_ends=     result.get("dead_ends", []),
        current_state= result.get("current_state", ""),
        next_steps=    result.get("next_steps", []),
        briefing=      result.get("briefing", ""),
        critic_score=  result.get("critic_score", 0),
    )


@app.get("/brains")
def list_brains(user_id: Optional[str] = None):
    return get_all_brains(user_id)


@app.get("/brains/{brain_id}")
def get_single_brain(brain_id: str):
    brain = get_brain(brain_id)
    if not brain:
        raise HTTPException(status_code=404, detail="Brain not found")
    return brain