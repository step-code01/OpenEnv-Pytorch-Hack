"""
models.py
Typed dataclasses for actions, observations, and state.
Shared by client.py and inference.py.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class DQAAction:
    action_type: str                   # query_stats | query_sample | flag_issue | submit
    column:      Optional[str] = None
    issue_type:  Optional[str] = None  # null_heavy | type_error | outlier

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class DQAObservation:
    query_result:    Dict[str, Any]
    steps_remaining: int
    flags_submitted: List[Dict]
    done:            bool
    reward:          float

    @classmethod
    def from_dict(cls, d: dict) -> "DQAObservation":
        return cls(
            query_result    = d.get("query_result", {}),
            steps_remaining = d.get("steps_remaining", 0),
            flags_submitted = d.get("flags_submitted", []),
            done            = d.get("done", False),
            reward          = d.get("reward", 0.0),
        )


@dataclass
class DQAState:
    episode_id:   Optional[str]
    step_count:   int
    task:         Optional[str]
    seed:         Optional[int]
    total_issues: Optional[int] = None
    flags_count:  int = 0
    done:         bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "DQAState":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
