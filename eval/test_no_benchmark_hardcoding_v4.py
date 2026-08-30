from __future__ import annotations

import pathlib
import unittest


class NoBenchmarkHardcodingV4Tests(unittest.TestCase):
    def test_production_code_has_no_benchmark_scenario_or_event_hardcoding(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[1]
        needles = (
            "mumbai_competitor_price_pressure",
            "bengaluru_inventory_shortage",
            "delhi_churn_cascade",
            "chennai_fengal",
            "fengal",
            "scenario_truth",
        )
        scanned = []
        for base in (root / "src", root / "config"):
            for path in base.rglob("*"):
                if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                scanned.append(path)
                for needle in needles:
                    self.assertNotIn(needle, text, f"benchmark leakage in {path}")
        self.assertTrue(scanned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
