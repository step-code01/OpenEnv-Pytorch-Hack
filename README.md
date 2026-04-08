# Dataset Quality Audit — OpenEnv Environment

An RL environment where an agent inspects a synthetic tabular dataset and must
identify planted data quality issues within a fixed query budget.

Inspired by the three-signal reward structure of the ECSSN paper (Jindal et al., 2022):
detection score × evidence quality × budget efficiency.

---

## Directory Layout

```
dataset-quality-env/
├── inference.py            # Root inference script (hackathon requirement)
├── openenv.yaml            # Environment metadata
├── models.py               # Shared typed dataclasses
├── client.py               # HTTP client (mirrors OpenEnv HTTPEnvClient)
├── README.md
└── server/
    ├── app.py              # FastAPI — /reset  /step  /state
    ├── environment.py      # Core episode logic
    ├── dataset_generator.py# Synthetic data + issue planting
    ├── grader.py           # Arithmetic reward (no LLM judge)
    ├── requirements.txt
    └── Dockerfile
```

---

## Tasks

| Task   | Columns | Issues | Budget |
|--------|---------|--------|--------|
| easy   | 8       | 3      | 15     |
| medium | 12      | 5      | 20     |
| hard   | 15      | 8      | 25     |

Issue types the agent must find:
- `null_heavy` — column has > 40 % null values
- `type_error` — column contains strings mixed with numerics
- `outlier`    — column contains an extreme value (9999.0)

---

## Reward

```
score = 0.70 × detection_score + 0.30 × budget_efficiency
```

- `detection_score` = confirmed flags / total planted issues
- `budget_efficiency` = steps remaining / max_steps
- Both components are pure arithmetic — no LLM grader anywhere.

---

## Running Locally

### 1. Start the server

```bash
cd server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Quick sanity check

```bash
# Should return {"message": "Episode started ...", ...}
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}' | python -m json.tool

# Query a column
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "query_stats", "column": "col_0"}' | python -m json.tool

# Submit
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit"}' | python -m json.tool
```

### 3. Run inference

```bash
# From repo root
export HF_TOKEN=hf_xxx
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
export SERVER_URL="http://localhost:8000"
export DQA_TASK="easy"

python inference.py
```

Expected stdout:
```
[START] task=easy env=dataset-quality-audit model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=query_stats(col_0) reward=0.00 done=false error=null
...
[END] success=true steps=12 score=0.720 rewards=0.00,0.00,...,0.72
```

### 4. Docker

```bash
cd server
docker build -t dqa-env .
docker run -p 8000:8000 dqa-env
```

---

## Deploying to HF Space

1. Create a new HF Space (Docker SDK).
2. Push the `server/` directory contents as the Space repo root
   (the Space root needs `app.py` + `Dockerfile` at its top level,
   or point the Space to `server/Dockerfile`).
3. Set Space to **public** so the validator can reach `/reset`.
4. Note your Space URL: `https://<your-space>.hf.space`

---

## Pre-submission Validation

```bash
pip install openenv-core
chmod +x validate-submission.sh
./validate-submission.sh https://<your-space>.hf.space .
```

All three checks must pass:
- [x] HF Space `/reset` returns 200
- [x] `docker build` succeeds
- [x] `openenv validate` passes

---

## What Is Mocked / TODO Before Final Submission

| Item | Status | Notes |
|------|--------|-------|
| `query_correlation` action | **MOCKED** | Returns `{"mocked": true}`. Wire up in `environment.py` `_query_correlation()` for medium/hard task evidence quality bonus. |
| Evidence quality signal | **MOCKED** | Reward uses only detection + budget. Add `evidence_quality` component (0.20 weight) once `query_correlation` is live — shrink detection weight to 0.50. |
| `openenv.yaml` full spec | **PARTIAL** | Add `observation_space` and `action_space` schema blocks once you confirm the openenv-core validator requires them. |
| HF Space URL | **TODO** | Fill in after deploying. |
| Multi-task inference loop | **TODO** | `inference.py` currently runs one task per invocation. Wrap in a loop over `["easy", "medium", "hard"]` if the validator expects all three in a single run. |
| Async client | **TODO** | Current client is sync `requests`. If OpenEnv requires async, swap to `httpx.AsyncClient` — model after the sample `MyEnvV4Env.from_docker_image()` pattern. |

---

## How `query_correlation` Should Work (wire-up guide)

Add to `environment.py`:

```python
def _query_correlation(self, col_a: str, col_b: str) -> dict:
    data_a = self._ep["dataset"].get(col_a)
    data_b = self._ep["dataset"].get(col_b)
    if data_a is None or data_b is None:
        return {"error": "Column not found."}

    # Compute Pearson correlation over non-null pairs
    pairs = [
        (a, b) for a, b in zip(data_a, data_b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float))
    ]
    if len(pairs) < 2:
        return {"correlation": None, "n_pairs": len(pairs)}

    n = len(pairs)
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    corr = round(num / den, 4) if den else 0.0
    return {"col_a": col_a, "col_b": col_b, "correlation": corr, "n_pairs": n}
```

Then add `action_type == "query_correlation"` dispatch in `step()` and expose it in the
`StepRequest` schema in `app.py` with a `column_b: Optional[str]` field.

---

## Reproducibility

Every episode is fully seeded via `dataset_generator.generate_dataset(task, seed)`.
The seed is returned in `GET /state` so judges can verify reproducibility.
Baseline: `python inference.py` with `DQA_SEED=42` must produce the same score every run.
