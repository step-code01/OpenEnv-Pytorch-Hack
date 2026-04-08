"""
inference.py
Dataset Quality Audit — OpenEnv inference script.

Strictly follows the [START] / [STEP] / [END] stdout format.
Uses OpenAI client as required by hackathon rules.

Environment variables:
  API_BASE_URL        LLM endpoint  (default: HF router)
  MODEL_NAME          Model id      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN / API_KEY  Auth token
  DQA_TASK            easy | medium | hard   (default: easy)
  DQA_SEED            Episode seed           (default: 42)
  SERVER_URL          env server URL         (default: http://localhost:8000)
"""

import os
import re
import json
import textwrap
from typing import Optional, List

from openai import OpenAI
from client import DatasetQualityClient
from models import DQAAction

# ── Config ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "") #hf token enough optional
SERVER_URL   = os.getenv("SERVER_URL",   "http://localhost:8000")

TASK_NAME  = os.getenv("DQA_TASK",  "easy")
SEED       = int(os.getenv("DQA_SEED", "42"))
BENCHMARK  = "dataset-quality-audit"

MAX_STEPS   = 40          # hard ceiling — env budget is lower, this is a safety net
TEMPERATURE = 0.2         # low temp for structured output
MAX_TOKENS  = 200

SUCCESS_THRESHOLD = 0.5   # score >= this → success=true

# ── Prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a data quality auditor. You receive a synthetic dataset and must
    find planted quality issues within a limited query budget.

    Issue types you can flag:
      null_heavy  — column has more than 40% null values
      type_error  — column contains non-numeric strings mixed with numbers
      outlier     — column contains an extreme numeric outlier (>6 std from mean)

    Available actions (respond with EXACTLY one per turn, no other text):
      query_stats(col_name)              — get null%, type counts, mean, std, min, max
      query_sample(col_name)             — see 5 random values from the column
      flag_issue(col_name, issue_type)   — flag a confirmed issue
      submit()                           — finalize your report

    Strategy:
      1. Run query_stats on columns to spot anomalies.
      2. Use query_sample when you suspect type_error or outlier.
      3. Flag only when you have evidence.
      4. submit() when done or budget is low.

    Respond with ONLY the action, e.g.:
      query_stats(col_3)
      flag_issue(col_7, null_heavy)
      submit()
""").strip()


def build_user_prompt(obs_dict: dict, history: List[str]) -> str:
    qr = obs_dict.get("query_result", {})
    steps = obs_dict.get("steps_remaining", 0)
    flags = obs_dict.get("flags_submitted", [])
    recent = "\n".join(history[-6:]) if history else "None"

    return textwrap.dedent(f"""
        Steps remaining: {steps}
        Flags so far: {json.dumps(flags)}
        Last result: {json.dumps(qr, default=str)}

        Recent history:
        {recent}

        Your next action:
    """).strip()


#action parse
_STATS_RE  = re.compile(r'query_stats\(\s*(["\']?)(\w+)\1\s*\)')
_SAMPLE_RE = re.compile(r'query_sample\(\s*(["\']?)(\w+)\1\s*\)')
_FLAG_RE   = re.compile(r'flag_issue\(\s*(["\']?)(\w+)\1\s*,\s*(["\']?)(\w+)\3\s*\)')
_SUBMIT_RE = re.compile(r'submit\(\s*\)')


def parse_action(raw: str) -> DQAAction:
    """Parse LLM output into a DQAAction. Falls back to submit() on failure."""
    raw = raw.strip()

    m = _STATS_RE.search(raw)
    if m:
        return DQAAction(action_type="query_stats", column=m.group(2))

    m = _SAMPLE_RE.search(raw)
    if m:
        return DQAAction(action_type="query_sample", column=m.group(2))

    m = _FLAG_RE.search(raw)
    if m:
        return DQAAction(action_type="flag_issue", column=m.group(2), issue_type=m.group(4))

    if _SUBMIT_RE.search(raw):
        return DQAAction(action_type="submit")

    return DQAAction(action_type="submit") #fallback


#logging
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

def main() -> None:
    llm    = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    client = DatasetQualityClient(base_url=SERVER_URL)

    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = client.reset(task=TASK_NAME, seed=SEED)
        obs_dict = {
            "query_result":    obs.query_result,
            "steps_remaining": obs.steps_remaining,
            "flags_submitted": obs.flags_submitted,
        }

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            user_prompt = build_user_prompt(obs_dict, history)
            raw_action  = "submit()"
            error_msg   = None

            try:
                completion = llm.chat.completions.create(
                    model       = MODEL_NAME,
                    messages    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature = TEMPERATURE,
                    max_tokens  = MAX_TOKENS,
                )
                raw_action = (completion.choices[0].message.content or "submit()").strip()
            except Exception as exc:
                error_msg  = str(exc)[:80]
                raw_action = "submit()"

            #Parse and step
            action = parse_action(raw_action)
            obs    = client.step(action)
            reward = obs.reward
            done   = obs.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=raw_action, reward=reward, done=done, error=error_msg)

            history.append(f"Step {step}: {raw_action} → reward={reward:.2f}")

            obs_dict = {
                "query_result":    obs.query_result,
                "steps_remaining": obs.steps_remaining,
                "flags_submitted": obs.flags_submitted,
            }

            if done:
                score = reward   # final reward IS the episode score
                break

        # In case episode ended without explicit done reward
        if rewards:
            score = max(rewards)

        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
