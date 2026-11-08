"""Persistent local request budgets for a sequential notebook run."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


class DailyBudgetExceeded(RuntimeError):
    """The configured local daily request budget has been used."""


class DailyRequestBudget:
    """Count attempts before sending, including failures and retries.

    Keys belong to separate projects for this assignment. This local ledger cannot
    see requests made outside the notebook or before tracking was introduced.
    Quota days follow Gemini's midnight Pacific reset. Do not share this ledger
    between concurrently running notebooks.
    """

    def __init__(self, path: str | Path, limit: int = 5):
        if type(limit) is not int or limit < 1:
            raise ValueError("Daily request limit must be a positive integer")

        self.path = Path(path)
        self.limit = limit

    @staticmethod
    def _today() -> str:
        return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()

    @staticmethod
    def _key_id(api_key: str) -> str:
        # Store fingerprints, never API keys. Aliases of the same key share usage.
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _load(self) -> dict:
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state["date"] == self._today():
                return state

        return {"date": self._today(), "timezone": "America/Los_Angeles", "keys": {}}

    def used(self, api_key: str) -> int:
        record = self._load()["keys"].get(self._key_id(api_key), {})
        return record.get("attempts", 0)

    def reserve(self, api_key: str, key_name: str) -> None:
        """Reserve one attempt before a network call; never charge cached results."""
        state = self._load()
        key_id = self._key_id(api_key)
        record = state["keys"].setdefault(key_id, {"name": key_name, "attempts": 0})

        if record["attempts"] >= self.limit:
            raise DailyBudgetExceeded(
                f"{key_name}: local daily budget exhausted ({self.limit} attempts)."
            )

        record["attempts"] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(self.path)
