"""
dataset_generator.py
Generates a synthetic tabular dataset with planted quality issues.
Pure Python — zero external dependencies.
"""
import random
import math


TASK_CONFIG = {
    "easy":   {"n_cols": 8,  "n_rows": 100, "n_issues": 3,  "max_steps": 15},
    "medium": {"n_cols": 12, "n_rows": 100, "n_issues": 5,  "max_steps": 20},
    "hard":   {"n_cols": 15, "n_rows": 100, "n_issues": 8,  "max_steps": 25},
}

ISSUE_TYPES = ["null_heavy", "type_error", "outlier"]


def _gauss(rng: random.Random, mu: float = 50.0, sigma: float = 10.0) -> float:
    return round(rng.gauss(mu, sigma), 2)


def generate_dataset(task: str, seed: int) -> dict:
    """
    Returns:
        {
          "columns": {"col_0": [v, ...], ...},   # col name → list of values
          "planted_issues": [
              {"column": "col_3", "issue_type": "null_heavy"},
              ...
          ],
          "column_names": ["col_0", ...],
          "n_rows": 100,
          "max_steps": 15,
        }
    """
    cfg = TASK_CONFIG[task]
    rng = random.Random(seed)

    n_cols  = cfg["n_cols"]
    n_rows  = cfg["n_rows"]
    n_issues = cfg["n_issues"]

    # --- Build clean columns ---
    col_names = [f"col_{i}" for i in range(n_cols)]
    columns: dict[str, list] = {
        name: [_gauss(rng) for _ in range(n_rows)]
        for name in col_names
    }

    # --- Plant issues into randomly chosen columns ---
    issue_cols = rng.sample(col_names, n_issues)
    planted_issues = []

    for i, col in enumerate(issue_cols):
        issue_type = ISSUE_TYPES[i % len(ISSUE_TYPES)]

        if issue_type == "null_heavy":
            # 45 % of values → None
            bad_idx = rng.sample(range(n_rows), int(n_rows * 0.45))
            for idx in bad_idx:
                columns[col][idx] = None

        elif issue_type == "type_error":
            # 10 values replaced with a non-numeric string
            bad_idx = rng.sample(range(n_rows), 10)
            for idx in bad_idx:
                columns[col][idx] = "N/A"

        elif issue_type == "outlier":
            # Single extreme value — clearly > 6 std from mean
            idx = rng.randint(0, n_rows - 1)
            columns[col][idx] = 9999.0

        planted_issues.append({"column": col, "issue_type": issue_type})

    return {
        "columns":        columns,
        "planted_issues": planted_issues,
        "column_names":   col_names,
        "n_rows":         n_rows,
        "max_steps":      cfg["max_steps"],
    }
