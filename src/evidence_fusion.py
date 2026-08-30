"""
Aletheia — src/evidence_fusion.py

Retrieval fusion is strictly downstream of deterministic evidence. External
text may only redistribute a bounded amount of existing relative hypothesis
weight among hypotheses that were actually assessed. It cannot create a new
driver, retrain a model, change a lag, or modify raw/mart data.
"""

from __future__ import annotations

import datetime as dt

from src import database

MAX_RETRIEVAL_INFLUENCE = 0.25
SUPPORT_MIN = -0.5
SUPPORT_MAX = 1.0


def reweight_and_renormalize(
    deterministic_scores: dict[str, float],
    retrieval_support: dict[str, float],
    lam: float = MAX_RETRIEVAL_INFLUENCE,
) -> dict[str, float]:
    """
    Apply bounded retrieval support WITHOUT changing unassessed hypotheses.

    For assessed hypotheses:
        raw_i = weight_i * (1 + lambda * support_i)

    The assessed hypotheses are then rescaled so their TOTAL pre-retrieval
    weight is preserved. Therefore retrieval can redistribute existing mass
    among the hypotheses it assessed, but cannot manufacture extra total mass.

    Important invariants:
    - no support entries -> exact no-op
    - all-zero support -> exact no-op
    - unassessed hypotheses -> exact no-op
    - assessed group total mass -> preserved
    - support is hard-bounded to [-0.5, 1.0]
    """
    if not (0 < lam <= MAX_RETRIEVAL_INFLUENCE):
        raise ValueError(
            f"lambda must be in (0, {MAX_RETRIEVAL_INFLUENCE}], got {lam}"
        )

    updated: dict[str, float] = {}
    for name, value in deterministic_scores.items():
        try:
            updated[name] = max(0.0, float(value))
        except (TypeError, ValueError):
            updated[name] = 0.0

    target_names = [
        name
        for name in updated
        if name in retrieval_support
        and abs(float(retrieval_support.get(name, 0.0) or 0.0)) > 1e-12
    ]
    if not target_names:
        return dict(updated)

    target_mass = sum(updated[name] for name in target_names)
    if target_mass <= 0:
        # Retrieval cannot create evidence from zero deterministic mass.
        return dict(updated)

    raw: dict[str, float] = {}
    for name in target_names:
        support = max(
            SUPPORT_MIN,
            min(SUPPORT_MAX, float(retrieval_support.get(name, 0.0) or 0.0)),
        )
        raw[name] = max(0.0, updated[name] * (1.0 + lam * support))

    raw_mass = sum(raw.values())
    if raw_mass <= 0:
        return dict(updated)

    scale = target_mass / raw_mass
    for name in target_names:
        updated[name] = raw[name] * scale

    return updated


def fuse_and_log(
    kpi: str,
    region: str,
    window_start: dt.date,
    deterministic_scores: dict[str, float],
    deterministic_probabilities: dict[str, float],
    retrieval_support: dict[str, float],
    retrieval_query: str | None,
) -> dict[str, float]:
    """
    Reweight stored deterministic relative hypothesis weights and audit every
    before/after value.

    deterministic_scores remains in the signature for backwards compatibility;
    deterministic_probabilities is the correct fusion basis because these are
    the weights shown to downstream consumers.
    """
    basis = deterministic_probabilities or deterministic_scores
    updated_probabilities = reweight_and_renormalize(basis, retrieval_support)

    names = list(dict.fromkeys([*basis.keys(), *deterministic_scores.keys()]))
    rows = []
    now = dt.datetime.utcnow()
    for driver_name in names:
        rows.append(
            (
                kpi,
                region,
                window_start,
                driver_name,
                float(basis.get(driver_name, 0.0)),
                retrieval_query,
                float(retrieval_support.get(driver_name, 0.0)),
                float(updated_probabilities.get(driver_name, basis.get(driver_name, 0.0))),
                now,
            )
        )

    sql = """
        INSERT INTO analysis.orchestrator_updates (
            kpi, region, window_start, driver_name,
            probability_before, retrieval_query, retrieval_support,
            probability_after, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    if rows:
        database.executemany(sql, rows)

    return updated_probabilities
