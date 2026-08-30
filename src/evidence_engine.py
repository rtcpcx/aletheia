"""
Aletheia — src/evidence_engine.py

The most important module in the system. Implements Stage 3 (deterministic
evidence) end to end:

    historical-only lag search -> lock lag -> historical Ridge regression
    -> minimum-N gate -> separate inferential validity (OLS/bootstrap, never
    fake Ridge p-values) -> incident severity z-score -> coefficient
    stability -> evidence score -> normalize -> calibrated softmax

No incident-window data is ever used to select a lag or fit a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.contracts import RootDriver
from src.guardrails_engine import (
    MIN_ABSOLUTE_OBSERVATIONS,
    MIN_OBSERVATIONS_PER_FEATURE,
    check_model_sufficiency,
)
from src.stationarity import is_usable_series, make_stationary

MIN_LAG_OVERLAP_OBSERVATIONS = 15
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
N_STABILITY_RESAMPLES = 20
STABILITY_RESAMPLE_FRACTION = 0.8
SOFTMAX_TEMPERATURE = 0.35
BOOTSTRAP_ITERATIONS = 500
SIGNIFICANCE_ALPHA = 0.05
RANDOM_SEED = 1337

CONSTANT_VARIANCE_EPS = 1e-9

EVIDENCE_HISTORICAL = "historical_relationship"
EVIDENCE_STRUCTURAL = "structural_break"
EVIDENCE_INSUFFICIENT = "insufficient_evidence"
LAG_CORRELATION_TIE_TOLERANCE = 0.02
MIN_COARSE_LAG_PERIODS = 6
HISTORICAL_Z_SATURATION = 3.0
DIRECTION_CONTRADICTION_WEIGHT = 0.10
MEDIATED_UPSTREAM_WEIGHT = 0.60


# ---------------------------------------------------------------------------
# 16.1 / 16.2 — Lag search and locking
# ---------------------------------------------------------------------------

def _coarse_period_frame(historical_driver: pd.Series, historical_component: pd.Series, historical_dates: pd.Series, cadence_days: int) -> pd.DataFrame:
    frame=pd.DataFrame({"metric_date":pd.to_datetime(historical_dates,errors="coerce"),"driver":pd.to_numeric(historical_driver,errors="coerce"),"component":pd.to_numeric(historical_component,errors="coerce")}).dropna(subset=["metric_date"])
    if frame.empty: return pd.DataFrame(columns=["driver","component"])
    cadence_days=max(1,int(cadence_days))
    if cadence_days==7:
        frame["source_period"]=frame["metric_date"].dt.to_period("W-SUN")
    else:
        origin=frame["metric_date"].min().normalize(); frame["source_period"]=(frame["metric_date"]-origin).dt.days//cadence_days
    return frame.groupby("source_period",sort=True)[["driver","component"]].mean()


def lag_search(historical_driver: pd.Series,historical_component: pd.Series,max_lag: int,*,historical_dates: pd.Series|None=None,source_cadence_days: int=1) -> tuple[int,float]:
    """Historical-only lag search with source-cadence semantics.

    Coarse sources are searched in whole source periods, not fake daily steps after
    forward-fill. Near-tied correlations keep the shorter lag.
    """
    cadence=max(1,int(source_cadence_days or 1))
    if cadence>1 and historical_dates is not None:
        f=_coarse_period_frame(historical_driver,historical_component,historical_dates,cadence)
        driver_series=f.get("driver",pd.Series(dtype=float)); component_series=f.get("component",pd.Series(dtype=float))
        max_steps=int(np.ceil(max_lag/cadence)) if max_lag>0 else 0
        candidates=[(step,step*cadence) for step in range(max_steps+1)]
        min_overlap=max(MIN_COARSE_LAG_PERIODS,int(np.ceil(MIN_LAG_OVERLAP_OBSERVATIONS/cadence)))
    else:
        driver_series=historical_driver; component_series=historical_component
        candidates=[(lag,lag) for lag in range(max_lag+1)]; min_overlap=MIN_LAG_OVERLAP_OBSERVATIONS
    best_lag=0; best_corr=0.0; best_abs=-1.0
    for shift_steps,lag_days in candidates:
        aligned=pd.concat([driver_series.shift(shift_steps),component_series],axis=1,keys=["driver","component"]).dropna()
        if len(aligned)<min_overlap: continue
        if aligned["driver"].std(ddof=0)==0 or aligned["component"].std(ddof=0)==0: continue
        corr=float(aligned["driver"].corr(aligned["component"]))
        if np.isnan(corr): continue
        if abs(corr)>best_abs+LAG_CORRELATION_TIE_TOLERANCE:
            best_abs=abs(corr); best_corr=corr; best_lag=lag_days
    return (0,0.0) if best_abs<0 else (int(best_lag),float(best_corr))


# ---------------------------------------------------------------------------
# 16.3 — Historical regression (Ridge, for stable relationship strength)
# ---------------------------------------------------------------------------

@dataclass
class RidgeFitResult:
    status: str  # "fitted" | "insufficient_history"
    coefficients: dict[str, float] = field(default_factory=dict)
    alpha: float | None = None
    n_observations: int = 0
    holdout_r2: float | None = None
    coefficient_stability: dict[str, float] = field(default_factory=dict)


def _build_lagged_design_matrix(
    component_series: pd.Series,
    driver_series: dict[str, pd.Series],
    locked_lags: dict[str, int],
) -> pd.DataFrame:
    frame = {"component": component_series}
    for name, series in driver_series.items():
        frame[name] = series.shift(locked_lags[name])
    df = pd.DataFrame(frame)
    return df


def fit_ridge_group(
    component: str,
    drivers: list[RootDriver],
    training_data: pd.DataFrame,
    locked_lags: dict[str, int],
) -> RidgeFitResult:
    """
    Train on a sufficiently long continuous historical window.

    Steps:
    1. apply previously selected (locked) lags
    2. align rows
    3. make usable series stationary
    4. remove invalid rows
    5. standardize features
    6. verify N is sufficient
    7. fit RidgeCV with alpha selected on historical time-series CV

    Returns coefficients, selected alpha, training observation count,
    holdout performance, and coefficient stability. If insufficient,
    returns status="insufficient_history" with no fabricated coefficients.
    """
    driver_names = [d.name for d in drivers]

    component_series, comp_diff = make_stationary(training_data[component])
    if not is_usable_series(component_series):
        return RidgeFitResult(status="insufficient_history", n_observations=0)

    driver_series: dict[str, pd.Series] = {}
    for name in driver_names:
        col = training_data[name]
        transformed, _ = make_stationary(col)
        driver_series[name] = transformed

    design = _build_lagged_design_matrix(component_series, driver_series, locked_lags)
    design = design.replace([np.inf, -np.inf], np.nan).dropna()

    n_features = len(driver_names)
    sufficiency = check_model_sufficiency(len(design), n_features)
    if sufficiency["status"] != "sufficient":
        return RidgeFitResult(status="insufficient_history", n_observations=len(design))

    X = design[driver_names].to_numpy()
    y = design["component"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_splits = min(5, max(2, len(design) // max(10, n_features * 5)))
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=n_splits)
    model.fit(X_scaled, y)

    coefficients = dict(zip(driver_names, model.coef_.tolist()))

    # Simple time-ordered holdout: last 20% of rows.
    split_idx = int(len(design) * 0.8)
    holdout_r2 = None
    if split_idx >= 10 and (len(design) - split_idx) >= 5:
        X_train, X_hold = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_hold = y[:split_idx], y[split_idx:]
        holdout_model = RidgeCV(alphas=RIDGE_ALPHAS, cv=max(2, min(5, split_idx // 10)))
        holdout_model.fit(X_train, y_train)
        holdout_r2 = float(holdout_model.score(X_hold, y_hold))

    stability = coefficient_stability(design, driver_names, model.alpha_)

    return RidgeFitResult(
        status="fitted",
        coefficients=coefficients,
        alpha=float(model.alpha_),
        n_observations=len(design),
        holdout_r2=holdout_r2,
        coefficient_stability=stability,
    )


# ---------------------------------------------------------------------------
# 16.4 — Correct significance handling (never fake Ridge p-values)
# ---------------------------------------------------------------------------

def inferential_validity(
    component: str,
    driver_name: str,
    training_data: pd.DataFrame,
    locked_lag: int,
    n_candidate_drivers: int,
) -> dict:
    """
    Ridge regression does not provide valid classical p-values. This
    function computes significance via a SEPARATE, single-driver, lag-locked
    OLS model when sample size/conditioning allow it, applies a Bonferroni
    correction across the candidate drivers tested for this component, and
    falls back to a bootstrap confidence interval when OLS assumptions
    cannot be trusted. If neither is possible, p_value is None.
    """
    component_series, _ = make_stationary(training_data[component])
    driver_series, _ = make_stationary(training_data[driver_name])

    shifted = driver_series.shift(locked_lag)
    aligned = pd.concat(
        [shifted, component_series], axis=1, keys=["driver", "component"]
    ).replace([np.inf, -np.inf], np.nan).dropna()

    n = len(aligned)
    if n < MIN_ABSOLUTE_OBSERVATIONS or aligned["driver"].std(ddof=0) == 0:
        return _bootstrap_validity(aligned, n_candidate_drivers)

    X = sm.add_constant(aligned["driver"].to_numpy())
    y = aligned["component"].to_numpy()

    try:
        ols_model = sm.OLS(y, X).fit()
    except Exception:
        return _bootstrap_validity(aligned, n_candidate_drivers)

    raw_p = float(ols_model.pvalues[1])
    corrected_p = min(1.0, raw_p * max(1, n_candidate_drivers))  # Bonferroni

    condition_number = float(np.linalg.cond(X))
    if condition_number > 30 or n < MIN_ABSOLUTE_OBSERVATIONS:
        return _bootstrap_validity(aligned, n_candidate_drivers)

    return {
        "method": "ols",
        "p_value": corrected_p,
        "is_significant": corrected_p < SIGNIFICANCE_ALPHA,
        "n": n,
    }


def _bootstrap_validity(aligned: pd.DataFrame, n_candidate_drivers: int) -> dict:
    if len(aligned) < MIN_LAG_OVERLAP_OBSERVATIONS or aligned.empty:
        return {"method": "none", "p_value": None, "is_significant": False, "n": len(aligned)}

    rng = np.random.default_rng(RANDOM_SEED)
    correlations = []
    n = len(aligned)
    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, size=n)
        sample = aligned.iloc[idx]
        if sample["driver"].std(ddof=0) == 0 or sample["component"].std(ddof=0) == 0:
            continue
        correlations.append(sample["driver"].corr(sample["component"]))

    if len(correlations) < 30:
        return {"method": "bootstrap", "p_value": None, "is_significant": False, "n": n}

    ci_low, ci_high = np.percentile(correlations, [2.5, 97.5])
    # Bonferroni-adjust the effective interval by widening the tail we check.
    alpha_adj = SIGNIFICANCE_ALPHA / max(1, n_candidate_drivers)
    lo_adj, hi_adj = np.percentile(
        correlations, [100 * alpha_adj / 2, 100 * (1 - alpha_adj / 2)]
    )
    is_significant = bool(lo_adj > 0 or hi_adj < 0)

    return {
        "method": "bootstrap",
        "p_value": None,
        "bootstrap_ci": (float(ci_low), float(ci_high)),
        "is_significant": is_significant,
        "n": n,
    }


# ---------------------------------------------------------------------------
# 16.5 — Incident severity
# ---------------------------------------------------------------------------

def compute_incident_zscore(
    incident_series: pd.Series, historical_reference: pd.Series
) -> float:
    """
    Robust (median/MAD-based) z-score measuring how unusual the incident
    driver movement is relative to historical baseline variation. This
    measures SEVERITY OF MOVEMENT, not historical causal strength.
    """
    ref = historical_reference.dropna()
    incident_value = incident_series.dropna()

    if ref.empty or incident_value.empty:
        return 0.0

    median = float(ref.median())
    mad = float((ref - median).abs().median())

    incident_mean = float(incident_value.mean())

    if mad < 1e-9:
        std = float(ref.std(ddof=0))
        if std < 1e-9:
            return 0.0
        return (incident_mean - float(ref.mean())) / std

    # 0.6745 scales MAD to be comparable to a standard deviation under
    # normality, so the robust z-score is on the same rough scale as a
    # classical one.
    robust_std = mad / 0.6745
    return (incident_mean - median) / robust_std



def compute_structural_break_score(
    baseline_series: pd.Series,
    incident_series: pd.Series,
) -> float:
    """
    Measure a level shift when the historical driver is effectively constant.

    Uses symmetric relative difference:

        |incident - baseline|
        ---------------------
        |incident| + |baseline|

    This is bounded approximately in [0, 1], scale-independent, and also
    handles zero-baseline event flags such as 0 -> 1.

    This is not a z-score and does not fabricate historical relationship
    evidence. It is used only for the explicit structural-break evidence
    regime when historical variance is effectively zero.
    """
    baseline = baseline_series.dropna()
    incident = incident_series.dropna()

    if baseline.empty or incident.empty:
        return 0.0

    baseline_value = float(baseline.mean())
    incident_value = float(incident.mean())

    delta = abs(incident_value - baseline_value)

    if delta < CONSTANT_VARIANCE_EPS:
        return 0.0

    denominator = (
        abs(baseline_value)
        + abs(incident_value)
        + CONSTANT_VARIANCE_EPS
    )

    return float(delta / denominator)


# ---------------------------------------------------------------------------
# 16.6 — Coefficient stability
# ---------------------------------------------------------------------------

def coefficient_stability(
    design: pd.DataFrame, driver_names: list[str], alpha: float
) -> dict[str, float]:
    """
    Fit the same lag-locked Ridge model repeatedly on bootstrap resamples of
    the historical training window and measure sign/scale consistency.
    A driver whose coefficient changes sign repeatedly gets a low score.

    Returns, per driver, the fraction of resamples where the coefficient
    sign matched the full-sample sign (1.0 = perfectly stable).
    """
    if design.empty:
        return {name: 0.0 for name in driver_names}

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(design)
    sample_size = max(int(n * STABILITY_RESAMPLE_FRACTION), MIN_ABSOLUTE_OBSERVATIONS)
    sample_size = min(sample_size, n)

    X_full = design[driver_names].to_numpy()
    y_full = design["component"].to_numpy()
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    from sklearn.linear_model import Ridge

    full_ridge = Ridge(alpha=alpha)
    full_ridge.fit(X_full_scaled, y_full)
    full_signs = np.sign(full_ridge.coef_)

    sign_matches = np.zeros(len(driver_names))

    for _ in range(N_STABILITY_RESAMPLES):
        idx = rng.choice(n, size=sample_size, replace=True)
        X_sample = X_full[idx]
        y_sample = y_full[idx]

        if np.std(y_sample) == 0:
            continue

        X_sample_scaled = scaler.transform(X_sample)
        resample_ridge = Ridge(alpha=alpha)
        try:
            resample_ridge.fit(X_sample_scaled, y_sample)
        except Exception:
            continue

        resample_signs = np.sign(resample_ridge.coef_)
        sign_matches += (resample_signs == full_signs).astype(float)

    stability_scores = sign_matches / N_STABILITY_RESAMPLES
    return dict(zip(driver_names, stability_scores.tolist()))


# ---------------------------------------------------------------------------
# 16.7 / 16.8 — Evidence score and normalization
# ---------------------------------------------------------------------------

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Convert valid non-negative scores to a stable common scale via
    score_i / max(valid_scores). Bounds the top score at 1.0 without
    artificially forcing the minimum to zero (unlike min-max scaling, which
    would exaggerate tiny differences between near-identical candidates).

    Handles all-zero vectors, a single candidate, and NaN entries.
    """
    scores = np.asarray(scores, dtype=float)
    valid_mask = ~np.isnan(scores)

    if not valid_mask.any():
        return np.zeros_like(scores)

    valid_scores = scores[valid_mask]
    max_score = np.max(valid_scores)

    if max_score <= 0:
        return np.where(valid_mask, 0.0, 0.0)

    normalized = np.where(valid_mask, np.clip(scores, 0, None) / max_score, 0.0)
    return normalized


def calibrated_softmax(
    normalized_scores: np.ndarray, temperature: float = SOFTMAX_TEMPERATURE
) -> np.ndarray:
    """
    Numerically stable softmax. Temperature must be > 0; it is configured
    and evaluated offline rather than chosen arbitrarily per call.

    The output is a RELATIVE HYPOTHESIS WEIGHT among the candidates
    presented, not a mathematically calibrated probability of causality.
    The dashboard must label it accordingly (see app.py).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    logits = np.asarray(normalized_scores, dtype=float) / temperature
    logits = np.where(np.isnan(logits), -np.inf, logits)

    if np.all(np.isneginf(logits)):
        return np.zeros_like(logits)

    stable_logits = logits - np.nanmax(logits[np.isfinite(logits)])
    exp_logits = np.where(np.isfinite(stable_logits), np.exp(stable_logits), 0.0)
    total = exp_logits.sum()
    if total <= 0:
        return np.zeros_like(logits)
    return exp_logits / total


@dataclass
class DriverEvidence:
    driver_name: str
    explains_component: str

    evidence_mode: str
    model_status: str

    best_lag_days: int

    baseline_value: float
    incident_value: float

    driver_zscore: float
    structural_break_score: float

    historical_coefficient: float
    holdout_correlation: float | None
    p_value: float | None
    is_significant: bool
    coefficient_stability: float

    evidence_score: float
    historical_lag_correlation: float = 0.0
    incident_severity_strength: float = 0.0
    direction_consistency: str = "not_evaluated"
    mechanism_weight: float = 1.0
    source_cadence_days: int = 1
    lag_resolution_days: int = 1
    causal_role: str = "direct"
    mediates_through: str | None = None
    expected_effect_sign: str = "unknown"
    normalized_score: float = 0.0
    softmax_probability: float = 0.0


def _lag_aligned_incident_series(
    *,
    incident_data: pd.DataFrame,
    incident_context_data: pd.DataFrame | None,
    driver_name: str,
    lag_days: int,
    source_cadence_days: int = 1,
) -> pd.Series:
    """Return driver values aligned to the KPI incident dates using the locked lag.

    If driver(t-lag) historically predicts component(t), then incident severity for
    the component window [T, T+n] must inspect the driver's [T-lag, T+n-lag]
    values. Falling back to the unshifted incident window would systematically miss
    legitimate leading indicators such as support tickets rising before churn.

    Lag selection itself remains historical-only; this helper only applies the already
    locked lag to incident interpretation.
    """
    if (
        incident_context_data is None
        or incident_context_data.empty
        or "metric_date" not in incident_data.columns
        or "metric_date" not in incident_context_data.columns
        or driver_name not in incident_context_data.columns
    ):
        return incident_data[driver_name].dropna() if driver_name in incident_data.columns else pd.Series(dtype=float)

    targets = pd.to_datetime(incident_data["metric_date"], errors="coerce").dropna()
    if targets.empty:
        return incident_data[driver_name].dropna() if driver_name in incident_data.columns else pd.Series(dtype=float)

    context = incident_context_data[["metric_date", driver_name]].copy()
    context["metric_date"] = pd.to_datetime(context["metric_date"], errors="coerce")
    context = context.dropna(subset=["metric_date"]).drop_duplicates("metric_date", keep="last")
    context = context.set_index("metric_date")[driver_name]

    source_dates = targets - pd.to_timedelta(int(lag_days), unit="D")
    values = context.reindex(source_dates).dropna()
    values.index = targets[: len(values)] if len(values) == len(targets) else values.index
    return values



def _bounded_incident_severity(zscore: float) -> float:
    return float(min(1.0,abs(float(zscore))/HISTORICAL_Z_SATURATION))

def _movement_sign(before: float,after: float) -> int:
    scale=max(abs(float(before)),abs(float(after)),1.0); delta=float(after)-float(before)
    if abs(delta)<=1e-6*scale: return 0
    return 1 if delta>0 else -1

def _direction_consistency(
    *,
    driver_baseline: float,
    driver_incident: float,
    component_baseline: float,
    component_incident: float,
    lag_correlation: float,
    expected_effect_sign: str = "unknown",
) -> tuple[str, float]:
    """Check incident direction using contract semantics before noisy model sign.

    Mediators and correlated drivers can make grouped regression or even historical
    pairwise signs unstable. When the KPI contract declares the expected business
    direction, that semantic sign is authoritative for coherence. Historical lag
    correlation is only a fallback for drivers whose effect direction is unknown.
    """
    ds = _movement_sign(driver_baseline, driver_incident)
    cs = _movement_sign(component_baseline, component_incident)
    if ds == 0 or cs == 0:
        return "not_evaluated", 1.0

    sign = str(expected_effect_sign or "unknown").strip().lower()
    if sign == "positive":
        effect_sign = 1
    elif sign == "negative":
        effect_sign = -1
    elif abs(lag_correlation) >= 1e-9:
        effect_sign = 1 if lag_correlation > 0 else -1
    else:
        return "not_evaluated", 1.0

    predicted_component_sign = ds * effect_sign
    return (
        ("aligned", 1.0)
        if predicted_component_sign == cs
        else ("contradictory", DIRECTION_CONTRADICTION_WEIGHT)
    )

def _mechanism_weight(driver: RootDriver,drivers: list[RootDriver]) -> float:
    if str(getattr(driver,"causal_role","direct"))!="upstream": return 1.0
    mediator=getattr(driver,"mediates_through",None)
    if mediator and any(d.name==mediator and d.explains==driver.explains for d in drivers): return MEDIATED_UPSTREAM_WEIGHT
    return 1.0

def compute_evidence_scores(kpi: str,region: str,component: str,drivers: list[RootDriver],historical_data: pd.DataFrame,incident_data: pd.DataFrame,incident_context_data: pd.DataFrame|None=None) -> list[DriverEvidence]:
    """Component-local evidence with cadence, direction and mechanism semantics."""
    if not drivers:
        return []
    dates=historical_data["metric_date"] if "metric_date" in historical_data.columns else None
    locked_lags={}; locked_corrs={}
    for d in drivers:
        lag,corr=lag_search(historical_data[d.name],historical_data[component],d.max_lag,historical_dates=dates,source_cadence_days=getattr(d,"source_cadence_days",1))
        locked_lags[d.name]=lag; locked_corrs[d.name]=corr
    fit=fit_ridge_group(component,drivers,historical_data,locked_lags)
    comp_hist=historical_data[component].dropna(); comp_inc=incident_data[component].dropna()
    comp_base=float(comp_hist.tail(30).mean()) if not comp_hist.empty else 0.0
    comp_incident=float(comp_inc.mean()) if not comp_inc.empty else 0.0
    results=[]; n_candidates=len(drivers)
    for d in drivers:
        lag=locked_lags[d.name]; corr=float(locked_corrs[d.name]); cadence=max(1,int(getattr(d,"source_cadence_days",1) or 1)); mech=_mechanism_weight(d,drivers)
        baseline_series=historical_data[d.name].dropna(); baseline=float(baseline_series.tail(30).mean()) if not baseline_series.empty else 0.0
        incident_series=_lag_aligned_incident_series(incident_data=incident_data,incident_context_data=incident_context_data,driver_name=d.name,lag_days=lag,source_cadence_days=cadence)
        incident=float(incident_series.mean()) if not incident_series.empty else 0.0
        z=compute_incident_zscore(incident_series,baseline_series); sev=_bounded_incident_severity(z)
        hist_std=float(baseline_series.std(ddof=0)) if not baseline_series.empty else 0.0
        sb=compute_structural_break_score(baseline_series,incident_series)
        has_sb=(not baseline_series.empty and hist_std<CONSTANT_VARIANCE_EPS and sb>CONSTANT_VARIANCE_EPS)
        common=dict(driver_name=d.name,explains_component=component,best_lag_days=lag,baseline_value=baseline,incident_value=incident,historical_lag_correlation=corr,source_cadence_days=cadence,lag_resolution_days=cadence,causal_role=str(getattr(d,"causal_role","direct")),mediates_through=getattr(d,"mediates_through",None),expected_effect_sign=str(getattr(d,"expected_effect_sign","unknown") or "unknown"),mechanism_weight=mech)
        if has_sb:
            consistency, dir_weight = _direction_consistency(
                driver_baseline=baseline, driver_incident=incident,
                component_baseline=comp_base, component_incident=comp_incident,
                lag_correlation=corr,
                expected_effect_sign=str(getattr(d,"expected_effect_sign","unknown") or "unknown"),
            )
            results.append(DriverEvidence(**common,evidence_mode=EVIDENCE_STRUCTURAL,model_status="historical_variance_unavailable",driver_zscore=0.0,structural_break_score=sb,historical_coefficient=0.0,holdout_correlation=None,p_value=None,is_significant=False,coefficient_stability=0.0,evidence_score=float(sb*dir_weight*mech),incident_severity_strength=float(sb),direction_consistency=consistency)); continue
        if fit.status!="fitted":
            results.append(DriverEvidence(**common,evidence_mode=EVIDENCE_INSUFFICIENT,model_status=fit.status,driver_zscore=z,structural_break_score=0.0,historical_coefficient=0.0,holdout_correlation=fit.holdout_r2,p_value=None,is_significant=False,coefficient_stability=0.0,evidence_score=0.0,incident_severity_strength=sev,direction_consistency="not_evaluated")); continue
        coef=float(fit.coefficients.get(d.name,0.0)); stability=float(fit.coefficient_stability.get(d.name,0.0))
        validity=inferential_validity(component=component,driver_name=d.name,training_data=historical_data,locked_lag=lag,n_candidate_drivers=n_candidates)
        validity_weight=1.0 if validity.get("is_significant") else 0.5
        consistency,dir_weight=_direction_consistency(driver_baseline=baseline,driver_incident=incident,component_baseline=comp_base,component_incident=comp_incident,lag_correlation=corr,expected_effect_sign=str(getattr(d,"expected_effect_sign","unknown") or "unknown"))
        relationship=min(1.0,abs(corr))
        score=relationship*sev*max(0.0,min(1.0,stability))*validity_weight*dir_weight*mech
        results.append(DriverEvidence(**common,evidence_mode=EVIDENCE_HISTORICAL,model_status=fit.status,driver_zscore=z,structural_break_score=0.0,historical_coefficient=coef,holdout_correlation=fit.holdout_r2,p_value=validity.get("p_value"),is_significant=bool(validity.get("is_significant",False)),coefficient_stability=stability,evidence_score=float(score),incident_severity_strength=sev,direction_consistency=consistency))
    raw=np.array([r.evidence_score for r in results],dtype=float); norm=normalize_scores(raw); probs=calibrated_softmax(norm,SOFTMAX_TEMPERATURE)
    for r,n,pv in zip(results,norm,probs): r.normalized_score=float(n); r.softmax_probability=float(pv)
    return results

