"""Append-only JSON Lines log of served predictions (read later by monitoring)."""

import json
from pathlib import Path


class PredictionLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, default=str) + "\n")
