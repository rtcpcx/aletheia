"""Regression guard: benchmark truth must not leak into production code."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = [ROOT / "src", ROOT / "config", ROOT / "app.py", ROOT / "refresh_decision_language.py"]

# Eval is allowed to know these. Production is not.
FORBIDDEN = (
    "scenario_truth",
    "mumbai_competitor_price_pressure",
    "bengaluru_inventory_shortage",
    "mumbai_heavy_rain_negative_control",
    "delhi_churn_cascade",
    "hyderabad_stale_source",
    "chennai_fengal_ambiguous_external",
    "fengal",
)


def _production_files() -> list[Path]:
    files: list[Path] = []
    for path in PRODUCTION_PATHS:
        if path.is_dir():
            files.extend(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".py", ".yaml", ".yml"}
            )
        elif path.is_file():
            files.append(path)
    return files


class NoBenchmarkHardcodingV3Tests(unittest.TestCase):
    def test_benchmark_truth_tokens_absent_from_production(self) -> None:
        hits: list[str] = []
        for path in _production_files():
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for token in FORBIDDEN:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)} contains {token!r}")
        self.assertEqual(hits, [], "benchmark leakage detected: " + "; ".join(hits))


if __name__ == "__main__":
    unittest.main(verbosity=2)
