"""
environment.py
DatasetQualityEnv — implements reset() / step() / state().
"""
import random
import statistics
from server.dataset_generator import generate_dataset, TASK_CONFIG
from server.grader import compute_reward, score_episode


class DatasetQualityEnv:
    """
    Single-agent environment.
    State is held in memory — one episode at a time (matches OpenEnv pattern).
    """

    def __init__(self):
        self._ep: dict | None = None  # episode state

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self, task: str = "easy", seed: int = 42) -> dict:
        """Start a new episode. Returns initial observation."""
        if task not in TASK_CONFIG:
            task = "easy"

        data = generate_dataset(task, seed)

        self._ep = {
            "task":            task,
            "seed":            seed,
            "dataset":         data["columns"],
            "planted_issues":  data["planted_issues"],
            "column_names":    data["column_names"],
            "n_rows":          data["n_rows"],
            "max_steps":       data["max_steps"],
            "flags":           [],
            "steps_used":      0,
            "done":            False,
            "final_score":     None,
        }

        return self._obs(
            query_result={
                "message":      "Episode started. Inspect the dataset.",
                "task":         task,
                "columns":      data["column_names"],
                "n_rows":       data["n_rows"],
                "budget":       data["max_steps"],
                "actions":      [
                    "query_stats(column)",
                    "query_sample(column)",
                    "flag_issue(column, issue_type)  # issue_type: null_heavy | type_error | outlier",
                    "submit()",
                ],
            }
        )

    def step(self, action: dict) -> dict:
        """Execute one action. Returns observation."""
        if self._ep is None:
            return self._obs({"error": "Call reset() first."}, reward=0.0, done=True)

        if self._ep["done"]:
            return self._obs({"error": "Episode already finished."}, reward=0.0, done=True)

        self._ep["steps_used"] += 1
        action_type = (action.get("action_type") or "submit").strip()

        # --- Dispatch ---
        if action_type == "query_stats":
            result = self._query_stats(action.get("column", ""))

        elif action_type == "query_sample":
            result = self._query_sample(action.get("column", ""))

        elif action_type == "flag_issue":
            result = self._flag_issue(
                action.get("column", ""),
                action.get("issue_type", ""),
            )

        elif action_type == "submit":
            result = self._submit()

        else:
            result = {
                "error": f"Unknown action_type {action_type!r}. "
                         "Use query_stats | query_sample | flag_issue | submit."
            }

        # Auto-terminate when budget runs out
        if self._ep["steps_used"] >= self._ep["max_steps"] and not self._ep["done"]:
            self._ep["done"] = True
            result["budget_exhausted"] = True #type: ignore

        # Compute reward only at episode end
        reward = 0.0
        if self._ep["done"] and self._ep["final_score"] is None:
            breakdown = score_episode(
                self._ep["flags"],
                self._ep["planted_issues"],
                self._ep["steps_used"],
                self._ep["max_steps"],
            )
            self._ep["final_score"] = breakdown["score"]
            result["score_breakdown"] = breakdown #type: ignore
            reward = breakdown["score"]

        return self._obs(result, reward=reward, done=self._ep["done"])

    def state(self) -> dict:
        """Return episode metadata (always available)."""
        if self._ep is None:
            return {
                "episode_id":     None,
                "step_count":     0,
                "task":           None,
                "seed":           None,
                "total_issues":   None,
                "flags_count":    0,
                "done":           False,
            }
        return {
            "episode_id":   f"{self._ep['task']}-seed{self._ep['seed']}",
            "step_count":   self._ep["steps_used"],
            "task":         self._ep["task"],
            "seed":         self._ep["seed"],
            "total_issues": len(self._ep["planted_issues"]),
            "flags_count":  len(self._ep["flags"]),
            "done":         self._ep["done"],
        }

    # ------------------------------------------------------------------ #
    #  Action handlers                                                     #
    # ------------------------------------------------------------------ #

    def _query_stats(self, column: str) -> dict:
        col_data = self._ep["dataset"].get(column) #type: ignore
        if col_data is None:
            return {"error": f"Column {column!r} not found.",
                    "available_columns": self._ep["column_names"]} #type: ignore

        n = len(col_data)
        nulls   = sum(1 for v in col_data if v is None)
        strings = sum(1 for v in col_data if isinstance(v, str))
        numeric = [v for v in col_data if isinstance(v, (int, float))]

        mean = round(sum(numeric) / len(numeric), 3) if numeric else None
        std  = round(statistics.stdev(numeric), 3) if len(numeric) > 1 else None
        mn   = min(numeric) if numeric else None
        mx   = max(numeric) if numeric else None

        return {
            "column":       column,
            "n_rows":       n,
            "null_pct":     round(nulls / n, 3),
            "string_count": strings,
            "mean":         mean,
            "std":          std,
            "min":          mn,
            "max":          mx,
        }

    def _query_sample(self, column: str) -> dict:
        col_data = self._ep["dataset"].get(column) #type: ignore
        if col_data is None:
            return {"error": f"Column {column!r} not found.",
                    "available_columns": self._ep["column_names"]} #type: ignore
        sample = random.sample(col_data, min(5, len(col_data)))
        return {"column": column, "sample": sample}

    def _flag_issue(self, column: str, issue_type: str) -> dict:
        valid_types = {"null_heavy", "type_error", "outlier"}
        if column not in self._ep["dataset"]: #type: ignore
            return {"error": f"Column {column!r} not found."}
        if issue_type not in valid_types:
            return {"error": f"issue_type must be one of {valid_types}."}

        # Deduplicate flags
        existing = any(
            f["column"] == column and f["issue_type"] == issue_type
            for f in self._ep["flags"] #type: ignore
        )
        if existing:
            return {"warning": f"Already flagged {column} as {issue_type}."}

        self._ep["flags"].append({"column": column, "issue_type": issue_type}) #type: ignore
        return {"flagged": column, "issue_type": issue_type,
                "total_flags": len(self._ep["flags"])} #type: ignore

    def _submit(self) -> dict:
        self._ep["done"] = True #type: ignore
        return {"submitted": True, "flags": self._ep["flags"]} #type: ignore

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _obs(self, query_result: dict, reward: float = 0.0, done: bool = False) -> dict:
        steps_remaining = 0
        flags = []
        if self._ep:
            steps_remaining = max(0, self._ep["max_steps"] - self._ep["steps_used"])
            flags = list(self._ep["flags"])
        return {
            "query_result":    query_result,
            "steps_remaining": steps_remaining,
            "flags_submitted": flags,
            "done":            done,
            "reward":          reward,
        }
