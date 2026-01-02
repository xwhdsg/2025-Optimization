from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt


def plot_from_json(
    json_path: str | Path,
    x_key: str,
    y_key: str,
    title: str,
    out_path: str | Path,
    label: Optional[str] = None,
) -> None:
    json_path = Path(json_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    xs = [r.get(x_key) for r in rows if x_key in r and y_key in r]
    ys = [r.get(y_key) for r in rows if x_key in r and y_key in r]

    plt.figure()
    plt.plot(xs, ys, label=label)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    if label is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
