"""
app.py
FastAPI server exposing the three OpenEnv endpoints:
  POST /reset
  POST /step
  GET  /state
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from environment import DatasetQualityEnv

app = FastAPI(title="Dataset Quality Audit Environment")
env = DatasetQualityEnv()


# ── Request schemas ────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"   # easy | medium | hard
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action_type: str               # query_stats | query_sample | flag_issue | submit
    column:      Optional[str] = None
    issue_type:  Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    return env.reset(task=req.task or "easy", seed=req.seed or 42)


@app.post("/step")
def step(req: StepRequest):
    return env.step(req.model_dump())


@app.get("/state")
def state():
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok"}
