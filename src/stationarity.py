"""
Aletheia — src/stationarity.py

Makes a series safe to feed into correlation/regression by testing for
stationarity (Augmented Dickey-Fuller) and first-differencing when needed.
Flat, too-short, or otherwise degenerate series are marked unusable rather
than silently passed downstream.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

MIN_USABLE_OBSERVATIONS = 12
ADF_ALPHA = 0.05


def _is_degenerate(series: pd.Series) -> bool:
    clean = series.dropna()
    if len(clean) < MIN_USABLE_OBSERVATIONS:
        return True
    if np.isinf(clean.to_numpy()).any():
        return True
    if np.isclose(clean.std(ddof=0), 0.0):
        return True
    return False


def make_stationary(series: pd.Series) -> tuple[pd.Series, bool]:
    """
    Run ADF.

    If p > 0.05:
        return first-differenced series, True

    Otherwise:
        return original series, False

    For flat, too-short, or invalid series:
        return safely transformed fallback and True, but the caller must
        check `is_usable_series` before regressing on the result — a
        constant-zero fallback is NOT sufficient evidence of stationarity,
        only a safe non-crashing default.
    """
    clean = series.dropna()

    if _is_degenerate(series):
        # Degenerate series cannot be meaningfully tested. Return a safe,
        # explicitly zeroed fallback so downstream code never receives NaN
        # or inf, and mark it as "differenced" (True) so callers treat it
        # with the same caution as a difference transform.
        fallback = pd.Series(
            np.zeros(len(series)), index=series.index, dtype=float
        )
        return fallback, True

    try:
        # ADF's autolag regression can become rank-deficient for highly smooth,
        # repeated, or cadence-expanded series. Treat that warning as an
        # unreliable ADF fit rather than printing it and using a dubious
        # stationarity result.
        with warnings.catch_warnings():
            warnings.simplefilter("error", SingularMatrixWarning)
            adf_stat, p_value, *_ = adfuller(
                clean.to_numpy(),
                autolag="AIC",
                result_object=False,
            )
    except SingularMatrixWarning:
        # Singular ADF design matrix => the test result is not uniquely
        # identified. Conservatively first-difference and let
        # is_usable_series() decide whether the transformed series may enter
        # downstream regression.
        differenced = series.diff().dropna()
        return differenced, True
    except Exception:
        # ADF itself can also fail on other pathological inputs. Use the same
        # safe fallback rather than allowing an invalid stationarity decision.
        differenced = series.diff().dropna()
        return differenced, True

    if p_value > ADF_ALPHA:
        differenced = series.diff().dropna()
        return differenced, True

    return series, False


def is_usable_series(series: pd.Series) -> bool:
    """
    Additional validation gate to run on the OUTPUT of make_stationary
    before it is allowed into a regression:

        minimum usable observations >= configured threshold
        no inf values
        no all-null output
        no zero variance after transformation
    """
    clean = series.dropna()
    if len(clean) < MIN_USABLE_OBSERVATIONS:
        return False
    if clean.empty:
        return False
    values = clean.to_numpy()
    if np.isinf(values).any():
        return False
    if np.isclose(np.nanstd(values, ddof=0), 0.0):
        return False
    return True
