"""
Refresh Aletheia's business recommendations and narration from existing analysis data.

This does NOT rerun changepoint detection, lag search, regression, decomposition, or
retrieval. It only rebuilds the presentation/action layer using already-computed RCA
results, plus a lightweight mart read to recover component baseline/incident direction.

Updated fields:
- decision_packets.recommended_action
- evidence_bundle.decision.action_context
- evidence_bundle.decision.recommended_action
- evidence_bundle.narration

Run from repo root after installing the updated action_engine.py, narrator.py and playbook:
    python refresh_decision_language.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import pandas as pd

from src import action_engine, database, guardrails_engine, narrator
from src.contracts import KpiContract, load_contracts
from src.driver_discovery import discover_candidates


MONITOR_CHANGE_THRESHOLD = 0.02


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return numeric if math.isfinite(numeric) else None


def _json_dumps_mysql(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _existing_bundle(kpi: str, region: str, window_start: dt.date) -> dict[str, Any]:
    df = database.query_df(
        """
        SELECT bundle_json
        FROM analysis.evidence_bundle
        WHERE kpi = %s AND region = %s AND window_start = %s
        """,
        params=(kpi, region, window_start),
    )
    if df.empty:
        return {}
    raw = df.iloc[0]["bundle_json"]
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if raw else {}
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _driver_rows(
    kpi: str,
    region: str,
    window_start: dt.date,
    *,
    contract: KpiContract,
    existing_bundle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Recover the ranked evidence without discarding V4 contract semantics.

    New V4 bundles persist cadence, direction-consistency and causal-role metadata
    inside deterministic_evidence.ranked_drivers. Prefer that richer representation.
    For legacy bundles, read the relational evidence table and enrich each driver from
    the KPI contract. The helper remains presentation-only: it does not recompute lags,
    models, z-scores or evidence weights.
    """
    bundle = existing_bundle or {}
    deterministic = bundle.get("deterministic_evidence")
    bundled_rows: list[dict[str, Any]] = []
    if isinstance(deterministic, dict):
        raw_rows = deterministic.get("ranked_drivers")
        if isinstance(raw_rows, list):
            bundled_rows = [dict(row) for row in raw_rows if isinstance(row, dict)]

    if bundled_rows:
        rows = bundled_rows
    else:
        df = database.query_df(
            """
            SELECT
                driver_name, explains_component, evidence_mode, model_status,
                best_lag_days, baseline_value, incident_value,
                structural_break_score, historical_coefficient, p_value,
                is_significant, coefficient_stability, driver_zscore,
                evidence_score, normalized_score, softmax_probability
            FROM analysis.driver_evidence
            WHERE kpi = %s AND region = %s AND window_start = %s
            """,
            params=(kpi, region, window_start),
        )
        if df.empty:
            return []
        rows = df.where(pd.notna(df), None).to_dict(orient="records")

    contract_by_name = {driver.name: driver for driver in contract.root_drivers}
    updates = database.query_df(
        """
        SELECT driver_name, probability_after, updated_at
        FROM analysis.orchestrator_updates
        WHERE kpi = %s AND region = %s AND window_start = %s
        ORDER BY updated_at
        """,
        params=(kpi, region, window_start),
    )
    fused: dict[str, float] = {}
    if not updates.empty:
        for _, update in updates.iterrows():
            if pd.notna(update.get("probability_after")):
                fused[str(update["driver_name"])] = float(update["probability_after"])

    for row in rows:
        name = str(row.get("driver_name") or "")
        spec = contract_by_name.get(name)
        if spec is not None:
            row.setdefault("source_cadence_days", int(getattr(spec, "source_cadence_days", 1) or 1))
            row.setdefault("lag_resolution_days", int(getattr(spec, "source_cadence_days", 1) or 1))
            row.setdefault("causal_role", str(getattr(spec, "causal_role", "direct") or "direct"))
            row.setdefault("mediates_through", getattr(spec, "mediates_through", None))
            row.setdefault("expected_effect_sign", str(getattr(spec, "expected_effect_sign", "unknown") or "unknown"))

        raw_weight = row.get("softmax_probability")
        row["softmax_probability"] = float(raw_weight) if raw_weight is not None else 0.0
        row["fused_probability"] = fused.get(name, float(row.get("fused_probability") or row["softmax_probability"]))
        row["is_significant"] = bool(row.get("is_significant"))
    return rows


def _decomposition(kpi: str, region: str, window_start: dt.date) -> dict[str, Any]:
    df = database.query_df(
        """
        SELECT decomposition_type, effect_a, effect_b, interaction_effect,
               residual, total_change, is_volatile, narrative_mode
        FROM analysis.pvm_decomposition
        WHERE kpi = %s AND region = %s AND window_start = %s
        """,
        params=(kpi, region, window_start),
    )
    if df.empty:
        return {}
    row = df.iloc[0]
    return {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}


def _component_impacts(contract: KpiContract, decomp: dict[str, Any]) -> dict[str, float]:
    if not contract.components:
        return {}
    if len(contract.components) == 1:
        return {contract.components[0].name: float(decomp.get("effect_a") or 0.0)}
    return {
        contract.components[0].name: float(decomp.get("effect_a") or 0.0),
        contract.components[1].name: float(decomp.get("effect_b") or 0.0),
    }


def _series_context(series: pd.Series, *, incident: bool) -> tuple[float | None, float | None]:
    clean = series.dropna()
    if clean.empty:
        return None, None
    if incident:
        half = max(1, len(clean) // 2)
        first = float(clean.iloc[:half].mean())
        second = float(clean.iloc[half:].mean() if len(clean) > half else clean.iloc[:half].mean())
        return first, second
    first = float(clean.iloc[: len(clean) // 2].mean()) if len(clean) > 1 else float(clean.mean())
    second = float(clean.iloc[len(clean) // 2 :].mean())
    return first, second


def _component_context(
    kpi: str,
    region: str,
    window_start: dt.date,
    contract: KpiContract,
    existing_bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover component baseline/incident levels without rerunning any ML.

    Prefer values already stored by a newer pipeline. For older bundles, perform only
    the same lightweight mart-window read used by driver discovery and reproduce the
    component summary convention from pipeline.py. No lag/model/evidence computation
    is run here.
    """
    decomp = existing_bundle.get("decomposition")
    if isinstance(decomp, dict):
        baselines = decomp.get("component_baselines")
        incidents = decomp.get("component_incidents")
        if isinstance(baselines, dict) and isinstance(incidents, dict) and baselines and incidents:
            return dict(baselines), dict(incidents)

    try:
        windows = discover_candidates(kpi, region, window_start, contract)
    except Exception:
        return {}, {}

    baselines: dict[str, Any] = {}
    incidents: dict[str, Any] = {}
    for component in contract.components:
        if component.name not in windows.model_training_window.columns:
            continue
        _, hist_end = _series_context(windows.model_training_window[component.name], incident=False)
        _, incident_end = _series_context(windows.incident_window[component.name], incident=True)
        baselines[component.name] = hist_end
        incidents[component.name] = incident_end
    return baselines, incidents


def _persist_packet_action(kpi: str, region: str, window_start: dt.date, recommendation: str) -> None:
    database.execute(
        """
        UPDATE analysis.decision_packets
        SET recommended_action = %s
        WHERE kpi = %s AND region = %s AND window_start = %s
        """,
        params=(recommendation, kpi, region, window_start),
    )


def _persist_bundle(kpi: str, region: str, window_start: dt.date, bundle: dict[str, Any]) -> None:
    database.execute(
        """
        REPLACE INTO analysis.evidence_bundle (
            kpi, region, window_start, bundle_json, generated_at
        ) VALUES (%s, %s, %s, %s, %s)
        """,
        params=(kpi, region, window_start, _json_dumps_mysql(bundle), dt.datetime.utcnow()),
    )


def main() -> None:
    contracts = load_contracts()
    packets = database.query_df(
        """
        SELECT kpi, region, window_start, percent_change, confidence_level,
               recommended_action, freshness_caveat
        FROM analysis.decision_packets
        ORDER BY kpi, region, window_start
        """
    )

    updated = 0
    skipped = 0
    for _, packet_row in packets.iterrows():
        kpi = str(packet_row["kpi"])
        region = str(packet_row["region"])
        window_start = pd.Timestamp(packet_row["window_start"]).date()
        contract = contracts.get(kpi)
        if contract is None:
            skipped += 1
            continue

        bundle = _existing_bundle(kpi, region, window_start)
        ranked = _driver_rows(
            kpi, region, window_start, contract=contract, existing_bundle=bundle
        )
        decomp = _decomposition(kpi, region, window_start)
        if not decomp:
            skipped += 1
            continue

        driver_names = [str(row.get("driver_name") or "") for row in ranked if row.get("driver_name")]
        required_sources = guardrails_engine.source_dependencies_for_analysis(kpi=kpi, driver_names=driver_names)
        source_health = guardrails_engine.check_source_health(
            region=region,
            as_of_date=window_start,
            source_names=required_sources,
        )

        existing_decision = bundle.get("decision", {}) if isinstance(bundle.get("decision"), dict) else {}

        pct_raw = packet_row.get("percent_change")
        pct = float(pct_raw) if pct_raw is not None and pd.notna(pct_raw) else None
        decision_type = str(existing_decision.get("decision_type") or "").strip().lower()
        if decision_type not in {"investigate", "monitor"}:
            decision_type = "investigate" if pct is None or abs(pct) > MONITOR_CHANGE_THRESHOLD else "monitor"

        component_impacts = _component_impacts(contract, decomp)
        component_baselines, component_incidents = _component_context(
            kpi, region, window_start, contract, bundle
        )

        action_context = action_engine.build_action_context(
            kpi=kpi,
            region=region,
            decision_type=decision_type,
            confidence=str(packet_row.get("confidence_level") or "Low"),
            ranked_drivers=ranked,
            source_health=source_health,
            component_impacts=component_impacts,
            component_baselines=component_baselines,
            component_incidents=component_incidents,
            complex_interaction=(
                str(decomp.get("narrative_mode") or "") == "complex_interaction"
                or bool(decomp.get("is_volatile"))
            ),
            kpi_relative_change=pct,
        )

        decision = dict(existing_decision)
        decision.update(
            {
                "kpi": kpi,
                "region": region,
                "window_start": str(window_start),
                "percent_change": pct,
                "confidence": str(packet_row.get("confidence_level") or "Low"),
                "confidence_level": str(packet_row.get("confidence_level") or "Low"),
                "decision_type": decision_type,
                "recommended_action": action_context["recommended_action"],
                "freshness_caveat": (
                    str(packet_row.get("freshness_caveat"))
                    if packet_row.get("freshness_caveat") is not None and pd.notna(packet_row.get("freshness_caveat"))
                    else None
                ),
                "source_health": source_health,
                "action_context": action_context,
            }
        )

        bundle.update(
            {
                "kpi": kpi,
                "region": region,
                "window_start": str(window_start),
                "deterministic_evidence": {"ranked_drivers": ranked},
                "decomposition": {
                    **decomp,
                    "component_impacts": component_impacts,
                    "component_baselines": component_baselines,
                    "component_incidents": component_incidents,
                },
                "confidence": {"level": decision["confidence_level"]},
                "decision": decision,
            }
        )

        narrations: dict[str, Any] = {}
        for persona in ("Executive", "Growth analyst"):
            narrations[persona] = narrator.narrate(bundle, persona)
        bundle["narration"] = narrations

        _persist_packet_action(kpi, region, window_start, action_context["recommended_action"])
        _persist_bundle(kpi, region, window_start, bundle)

        updated += 1
        print(
            f"UPDATED {kpi}/{region}/{window_start}: "
            f"{action_context.get('action_level')} | "
            f"{action_context.get('primary_driver_label') or 'no direct driver'}"
        )

    print(f"\nUpdated {updated} decision/narration bundle(s); skipped {skipped}.")


if __name__ == "__main__":
    main()
