from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class RunLogger:
    name: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    def log(self, **kwargs: Any) -> None:
        row = dict(kwargs)
        row.setdefault("t", time.time() - self.t0)
        self.rows.append(row)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2)
