"""
grader.py
Computes episode reward from flags, ground truth, and budget used.
Pure arithmetic — deterministic, no LLM judge.

Reward structure (ECSSN three-signal analogy):
  detection_score   (0.70 weight) — issues correctly identified
  budget_efficiency (0.30 weight) — steps remaining at submit time

Combined reward always in [0.0, 1.0].
"""


def _match(flag: dict, issues: list) -> bool:
    """True if flag matches any planted issue exactly."""
    for issue in issues:
        if (flag.get("column") == issue["column"] and
                flag.get("issue_type") == issue["issue_type"]):
            return True
    return False


def compute_reward(
    flags: list,
    planted_issues: list,
    steps_used: int,
    max_steps: int,
) -> float:
    """
    Args:
        flags:          list of {"column": str, "issue_type": str} submitted by agent
        planted_issues: ground truth from dataset_generator
        steps_used:     steps consumed this episode
        max_steps:      budget for this task

    Returns:
        float in [0.0, 1.0]
    """
    if not planted_issues:
        return 0.0

    # --- Signal 1: detection ---
    confirmed = sum(1 for f in flags if _match(f, planted_issues))
    detection_score = confirmed / len(planted_issues)

    # --- Signal 2: budget efficiency ---
    budget_efficiency = max(0.0, (max_steps - steps_used) / max_steps)

    # --- Combined ---
    reward = 0.70 * detection_score + 0.30 * budget_efficiency
    return round(min(max(reward, 0.0), 1.0), 4)


def score_episode(flags: list, planted_issues: list, steps_used: int, max_steps: int) -> dict:
    """Returns a breakdown dict — useful for debugging and [END] logging."""
    confirmed = sum(1 for f in flags if _match(f, planted_issues))
    false_positives = len(flags) - confirmed
    detection_score = confirmed / len(planted_issues) if planted_issues else 0.0
    budget_efficiency = max(0.0, (max_steps - steps_used) / max_steps)
    final = compute_reward(flags, planted_issues, steps_used, max_steps)

    return {
        "total_issues":     len(planted_issues),
        "confirmed":        confirmed,
        "false_positives":  false_positives,
        "detection_score":  round(detection_score, 4),
        "budget_efficiency": round(budget_efficiency, 4),
        "score":            final,
    }
