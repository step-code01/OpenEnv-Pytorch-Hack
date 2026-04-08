"""
client.py
HTTP client for the Dataset Quality Audit environment.
Mirrors the OpenEnv HTTPEnvClient pattern.
"""
import requests
from models import DQAAction, DQAObservation, DQAState


class DatasetQualityClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task: str = "easy", seed: int = 42) -> DQAObservation:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task": task, "seed": seed},
            timeout=30,
        )
        resp.raise_for_status()
        return DQAObservation.from_dict(resp.json())

    def step(self, action: DQAAction) -> DQAObservation:
        resp = requests.post(
            f"{self.base_url}/step",
            json=action.to_dict(),
            timeout=30,
        )
        resp.raise_for_status()
        return DQAObservation.from_dict(resp.json())

    def state(self) -> DQAState:
        resp = requests.get(f"{self.base_url}/state", timeout=30)
        resp.raise_for_status()
        return DQAState.from_dict(resp.json())
