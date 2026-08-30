"""
Aletheia — data/generate_benchmark_data.py

Generate the Aletheia Business RCA Benchmark v1.0.

The benchmark is semi-synthetic:
- Internal business data are synthetic and generated from known causal mechanisms.
- Selected external-event anchors use real geography/dates so Stage 4 can perform
  meaningful public-web retrieval.
- Ground-truth files are written only under data/generated/<split>/ and must never
  be imported by src/* or app.py.

Default output:
    data/generated/dev/
    data/generated/holdout/

Usage:
    python data/generate_benchmark_data.py
    python data/generate_benchmark_data.py --split dev
    python data/generate_benchmark_data.py --split holdout
    python data/generate_benchmark_data.py --output-root data/generated

The numerical business data are reproducible by seed. source_loaded_at defaults to
the current generation timestamp because it represents benchmark ingestion time,
not the historical business-event timestamp.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


START_DATE = dt.date(2024, 1, 1)
END_DATE = dt.date(2025, 2, 28)
REGIONS = ["Mumbai", "Bengaluru", "Delhi", "Chennai", "Hyderabad"]
MATURE_PRODUCTS = ["aurora", "nova", "pulse", "vertex"]
ALL_PRODUCTS = MATURE_PRODUCTS + ["orbit_new"]
CHANNELS = ["search", "social", "affiliate", "email"]

SEEDS = {"dev": 202610, "holdout": 202611}
ORBIT_LAUNCH_DATE = dt.date(2025, 1, 15)

COMPETITOR_DATE = dt.date(2024, 4, 15)
COMPANY_PRICE_RESPONSE_DATE = COMPETITOR_DATE + dt.timedelta(days=2)
INVENTORY_SHORTAGE_DATE = dt.date(2024, 6, 10)
MUMBAI_RAIN_DATE = dt.date(2024, 7, 8)
DELHI_SUPPORT_SPIKE_DATE = dt.date(2024, 8, 12)
DELHI_UPTIME_DIP_DATE = dt.date(2024, 8, 16)
DELHI_CHURN_DATE = dt.date(2024, 8, 19)
HYDERABAD_STALE_START = dt.date(2024, 9, 23)
HYDERABAD_STALE_END = dt.date(2024, 9, 25)
DIWALI_DATE = dt.date(2024, 10, 31)
CHENNAI_CAMPAIGN_START = dt.date(2024, 11, 28)
CHENNAI_FENGAL_DATE = dt.date(2024, 11, 30)
CHENNAI_FENGAL_END = dt.date(2024, 12, 6)

PRODUCTS = {
    "aurora": {
        "price": 118.0,
        "unit_cost": 66.0,
        "daily_demand": 48.0,
        "reorder_point": 95,
    },
    "nova": {
        "price": 82.0,
        "unit_cost": 43.0,
        "daily_demand": 72.0,
        "reorder_point": 135,
    },
    "pulse": {
        "price": 64.0,
        "unit_cost": 31.0,
        "daily_demand": 88.0,
        "reorder_point": 160,
    },
    "vertex": {
        "price": 146.0,
        "unit_cost": 83.0,
        "daily_demand": 37.0,
        "reorder_point": 75,
    },
    "orbit_new": {
        "price": 134.0,
        "unit_cost": 72.0,
        "daily_demand": 18.0,
        "reorder_point": 40,
    },
}

REGION_FACTOR = {
    "Mumbai": 1.18,
    "Bengaluru": 1.12,
    "Delhi": 1.10,
    "Chennai": 0.98,
    "Hyderabad": 0.94,
}

CHANNEL_BASE_SPEND = {
    "search": 1350.0,
    "social": 1050.0,
    "affiliate": 620.0,
    "email": 330.0,
}
CHANNEL_SESSION_EFF = {
    "search": 0.82,
    "social": 0.68,
    "affiliate": 0.76,
    "email": 1.18,
}
CHANNEL_CVR_MULT = {
    "search": 1.08,
    "social": 0.88,
    "affiliate": 1.02,
    "email": 1.18,
}


def date_range(start: dt.date = START_DATE, end: dt.date = END_DATE):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def week_start(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def within(d: dt.date, start: dt.date, end: dt.date) -> bool:
    return start <= d <= end


def weekday_factor(d: dt.date) -> float:
    # Consumer demand / sessions are moderately higher Fri-Sun.
    return [0.93, 0.96, 0.98, 1.00, 1.06, 1.13, 1.10][d.weekday()]


def smooth_annual_factor(d: dt.date) -> float:
    doy = d.timetuple().tm_yday
    return 1.0 + 0.035 * math.sin(2.0 * math.pi * (doy - 35) / 365.25)


def trend_factor(d: dt.date) -> float:
    days = (d - START_DATE).days
    return 1.0 + 0.00032 * days


def diwali_factor(d: dt.date) -> float:
    # Short, expected seasonal effect — deliberately not a long structural shift.
    distance = abs((d - DIWALI_DATE).days)
    if distance > 4:
        return 1.0
    return 1.0 + 0.12 * (1.0 - distance / 5.0)


def ar1_series(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, sigma)
    return x


@dataclass
class SplitParameters:
    competitor_drop: float
    company_price_response: float
    bengaluru_replenishment_fraction: float
    chennai_replenishment_fraction: float
    chennai_weather_demand_factor: float
    chennai_campaign_multiplier: float
    delhi_support_multiplier: float
    delhi_churn_multiplier: float


def split_parameters(rng: np.random.Generator) -> SplitParameters:
    # Narrow seeded jitter prevents hand-tuning to one exact deterministic magnitude.
    return SplitParameters(
        competitor_drop=float(rng.uniform(0.865, 0.895)),
        company_price_response=float(rng.uniform(0.915, 0.935)),
        bengaluru_replenishment_fraction=float(rng.uniform(0.38, 0.46)),
        chennai_replenishment_fraction=float(rng.uniform(0.55, 0.64)),
        chennai_weather_demand_factor=float(rng.uniform(0.84, 0.90)),
        chennai_campaign_multiplier=float(rng.uniform(1.24, 1.32)),
        delhi_support_multiplier=float(rng.uniform(1.55, 1.72)),
        delhi_churn_multiplier=float(rng.uniform(1.42, 1.58)),
    )


def build_market_context(
    rng: np.random.Generator,
    params: SplitParameters,
    loaded_at: dt.datetime,
) -> pd.DataFrame:
    weeks = sorted({week_start(d) for d in date_range()})
    rows: list[dict[str, Any]] = []

    region_noise: dict[str, np.ndarray] = {
        r: ar1_series(rng, len(weeks), phi=0.55, sigma=0.007) for r in REGIONS
    }
    weather_noise: dict[str, np.ndarray] = {
        r: ar1_series(rng, len(weeks), phi=0.35, sigma=0.025) for r in REGIONS
    }

    for region in REGIONS:
        for i, w in enumerate(weeks):
            competitor = 1.0 + region_noise[region][i]
            weather = 1.0 + weather_noise[region][i]
            event_type = None

            if region == "Mumbai" and w >= week_start(COMPETITOR_DATE):
                competitor *= params.competitor_drop

            # Real-world external negative control: signal exists, business DGP
            # intentionally applies no causal impact from this event.
            if region == "Mumbai" and w == week_start(MUMBAI_RAIN_DATE):
                weather = 0.52
                event_type = "heavy_rain_external_control"

            # Real externally resolvable ambiguity scenario. Weekly context creates
            # a realistic cadence mismatch with daily business facts.
            if region == "Chennai" and w in {
                week_start(CHENNAI_FENGAL_DATE),
                week_start(CHENNAI_FENGAL_DATE + dt.timedelta(days=7)),
            }:
                weather = 0.58 if w == week_start(CHENNAI_FENGAL_DATE) else 0.68
                event_type = "severe_weather_disruption"

            rows.append(
                {
                    "week_start": w,
                    "region": region,
                    "competitor_price_index": round(clamp(competitor, 0.70, 1.15), 4),
                    "weather_index": round(clamp(weather, 0.35, 1.20), 4),
                    "event_type": event_type,
                    "source_type": "scheduled_feed",
                    "source_loaded_at": loaded_at,
                }
            )

    return pd.DataFrame(rows)


def daily_context_maps(context: pd.DataFrame):
    comp = {}
    weather = {}
    event = {}
    for row in context.itertuples(index=False):
        for offset in range(7):
            d = row.week_start + dt.timedelta(days=offset)
            if START_DATE <= d <= END_DATE:
                key = (row.region, d)
                comp[key] = float(row.competitor_price_index)
                weather[key] = float(row.weather_index)
                event[key] = row.event_type
    return comp, weather, event


def build_marketing(
    rng: np.random.Generator,
    params: SplitParameters,
    competitor_daily: dict[tuple[str, dt.date], float],
    loaded_at: dt.datetime,
) -> pd.DataFrame:
    days = list(date_range())
    region_ar = {r: ar1_series(rng, len(days), phi=0.62, sigma=0.025) for r in REGIONS}
    rows: list[dict[str, Any]] = []

    for region in REGIONS:
        base_cvr = {
            "Mumbai": 0.071,
            "Bengaluru": 0.074,
            "Delhi": 0.068,
            "Chennai": 0.070,
            "Hyderabad": 0.072,
        }[region]

        for i, d in enumerate(days):
            # Explicit 3-day feed outage: rows are absent, not silently zero-filled.
            if region == "Hyderabad" and within(d, HYDERABAD_STALE_START, HYDERABAD_STALE_END):
                continue

            seasonal = weekday_factor(d) * smooth_annual_factor(d) * trend_factor(d)
            diwali = diwali_factor(d)

            for channel in CHANNELS:
                spend = (
                    CHANNEL_BASE_SPEND[channel]
                    * REGION_FACTOR[region]
                    * trend_factor(d)
                    * (1.0 + 0.35 * region_ar[region][i])
                    * rng.lognormal(mean=0.0, sigma=0.055)
                )

                # Chennai campaign deliberately pushes demand upward while Fengal
                # pushes fulfillment/conversion downward.
                if region == "Chennai" and within(
                    d, CHENNAI_CAMPAIGN_START, CHENNAI_FENGAL_END
                ):
                    spend *= params.chennai_campaign_multiplier

                if within(d, DIWALI_DATE - dt.timedelta(days=4), DIWALI_DATE + dt.timedelta(days=4)):
                    spend *= 1.06

                impressions = max(
                    100,
                    int(
                        round(
                            spend
                            * {
                                "search": 26,
                                "social": 47,
                                "affiliate": 22,
                                "email": 64,
                            }[channel]
                            * rng.normal(1.0, 0.035)
                        )
                    ),
                )

                sessions = max(
                    1,
                    int(
                        round(
                            spend
                            * CHANNEL_SESSION_EFF[channel]
                            * seasonal
                            * diwali
                            * rng.normal(1.0, 0.035)
                        )
                    ),
                )

                cvr = base_cvr * CHANNEL_CVR_MULT[channel]
                cvr *= 1.0 + 0.07 * region_ar[region][i]

                competitor = competitor_daily[(region, d)]
                if region == "Mumbai" and d >= COMPANY_PRICE_RESPONSE_DATE:
                    # Competitor pressure remains a real conversion headwind even
                    # after the business reacts on price.
                    cvr *= 0.90

                if region == "Chennai" and within(d, CHENNAI_FENGAL_DATE, CHENNAI_FENGAL_END):
                    cvr *= 0.82

                cvr = clamp(cvr, 0.025, 0.14)
                orders = max(0, int(round(sessions * cvr * rng.normal(1.0, 0.025))))
                new_customers = max(
                    0,
                    min(
                        orders,
                        int(round(orders * rng.uniform(0.57, 0.69))),
                    ),
                )

                rows.append(
                    {
                        "activity_date": d,
                        "region": region,
                        "channel": channel,
                        "spend": round(float(spend), 2),
                        "impressions": impressions,
                        "sessions": sessions,
                        "orders": orders,
                        "new_customers": new_customers,
                        "source_loaded_at": loaded_at,
                    }
                )

    return pd.DataFrame(rows)


def marketing_daily_spend(marketing: pd.DataFrame) -> dict[tuple[str, dt.date], float]:
    grouped = (
        marketing.groupby(["region", "activity_date"], as_index=False)["spend"]
        .sum()
    )
    return {
        (row.region, row.activity_date): float(row.spend)
        for row in grouped.itertuples(index=False)
    }


def build_sales(
    rng: np.random.Generator,
    params: SplitParameters,
    marketing_spend: dict[tuple[str, dt.date], float],
    loaded_at: dt.datetime,
) -> pd.DataFrame:
    days = list(date_range())
    region_ar = {r: ar1_series(rng, len(days), phi=0.72, sigma=0.022) for r in REGIONS}
    rows: list[dict[str, Any]] = []

    # Opening inventory state per region/product.
    inventory: dict[tuple[str, str], float] = {}
    for region in REGIONS:
        for product in MATURE_PRODUCTS:
            spec = PRODUCTS[product]
            inventory[(region, product)] = spec["reorder_point"] * 2.25

    # orbit_new is intentionally launched only in Bengaluru.
    inventory[("Bengaluru", "orbit_new")] = PRODUCTS["orbit_new"]["reorder_point"] * 2.0

    # Expected baseline regional spend allows marketing pressure to enter sales DGP.
    baseline_region_spend = {
        region: sum(CHANNEL_BASE_SPEND.values()) * REGION_FACTOR[region]
        for region in REGIONS
    }

    for region in REGIONS:
        for i, d in enumerate(days):
            product_list = list(MATURE_PRODUCTS)
            if region == "Bengaluru" and d >= ORBIT_LAUNCH_DATE:
                product_list.append("orbit_new")

            actual_spend = marketing_spend.get((region, d))
            if actual_spend is None:
                # Underlying demand exists during the Hyderabad feed outage; missing
                # marketing rows are an observability defect, not a business shock.
                marketing_multiplier = 1.0
            else:
                spend_ratio = actual_spend / baseline_region_spend[region]
                marketing_multiplier = 1.0 + 0.18 * (spend_ratio - 1.0)

            common = (
                REGION_FACTOR[region]
                * weekday_factor(d)
                * smooth_annual_factor(d)
                * trend_factor(d)
                * diwali_factor(d)
                * marketing_multiplier
                * (1.0 + region_ar[region][i])
            )

            # Mumbai competitor pressure reduces quantity demand modestly.
            if region == "Mumbai" and d >= COMPANY_PRICE_RESPONSE_DATE:
                common *= 0.94

            # Chennai severe weather is a real business intervention in this
            # benchmark (unlike the Mumbai July negative control).
            if region == "Chennai" and within(d, CHENNAI_FENGAL_DATE, CHENNAI_FENGAL_END):
                common *= params.chennai_weather_demand_factor

            for product in product_list:
                spec = PRODUCTS[product]
                key = (region, product)

                if product == "orbit_new":
                    age = max(0, (d - ORBIT_LAUNCH_DATE).days)
                    launch_ramp = min(1.0, 0.28 + age / 35.0)
                else:
                    launch_ramp = 1.0

                product_noise = rng.normal(1.0, 0.045)
                desired = (
                    spec["daily_demand"] * common * launch_ramp * product_noise
                )

                # Company price response to competitor movement begins exactly two
                # days after the externally visible competitor regime change.
                list_price = spec["price"]
                if region == "Mumbai" and d >= COMPANY_PRICE_RESPONSE_DATE:
                    list_price *= params.company_price_response
                list_price *= rng.normal(1.0, 0.006)

                discount = 0.038 + rng.normal(0.0, 0.008)
                if within(d, DIWALI_DATE - dt.timedelta(days=3), DIWALI_DATE + dt.timedelta(days=3)):
                    discount += 0.012
                discount = clamp(discount, 0.005, 0.12)

                opening_inventory = inventory[key]

                # Structural Bengaluru supply shortage.
                if region == "Bengaluru" and d == INVENTORY_SHORTAGE_DATE:
                    opening_inventory = min(
                        opening_inventory,
                        spec["reorder_point"] * 0.62,
                    )
                bengaluru_shortage = (
                    region == "Bengaluru"
                    and INVENTORY_SHORTAGE_DATE <= d <= dt.date(2024, 7, 5)
                    and product in MATURE_PRODUCTS
                )

                # Chennai weather/logistics shortage competes with the simultaneous
                # marketing campaign in the ambiguous Stage-4 scenario.
                # Only the two highest-volume SKUs become supply constrained.
                # This keeps region-level stock availability around 0.5 rather
                # than collapsing to zero, making weather and stock evidence
                # intentionally competitive in the Stage-4 ambiguity case.
                chennai_constrained_products = {"nova", "pulse"}
                if (
                    region == "Chennai"
                    and product in chennai_constrained_products
                    and d == CHENNAI_FENGAL_DATE
                ):
                    opening_inventory = min(
                        opening_inventory,
                        spec["reorder_point"] * 0.68,
                    )
                chennai_shortage = (
                    region == "Chennai"
                    and within(d, CHENNAI_FENGAL_DATE, CHENNAI_FENGAL_END)
                    and product in chennai_constrained_products
                )

                desired_units = max(0, int(round(desired)))
                units_sold = min(desired_units, max(0, int(math.floor(opening_inventory))))

                unit_cost = spec["unit_cost"] * rng.normal(1.0, 0.004)
                net_revenue = units_sold * list_price * (1.0 - discount)

                # Store opening inventory because mart.stock_availability evaluates
                # whether inventory was adequate when demand was faced.
                rows.append(
                    {
                        "sale_date": d,
                        "region": region,
                        "product_id": product,
                        "units_sold": units_sold,
                        "list_price": round(float(list_price), 2),
                        "unit_cost": round(float(unit_cost), 2),
                        "discount_pct": round(float(discount), 4),
                        "net_revenue": round(float(net_revenue), 2),
                        "inventory_units": max(0, int(round(opening_inventory))),
                        "reorder_point": int(spec["reorder_point"]),
                        "source_loaded_at": loaded_at,
                    }
                )

                normal_replenishment = spec["daily_demand"] * REGION_FACTOR[region]
                replenishment = normal_replenishment * rng.normal(1.0, 0.035)

                if bengaluru_shortage:
                    replenishment *= params.bengaluru_replenishment_fraction
                if chennai_shortage:
                    replenishment *= params.chennai_replenishment_fraction

                inventory[key] = max(
                    0.0,
                    opening_inventory - units_sold + max(0.0, replenishment),
                )

                # Normal operations use an inventory-order-up policy. During supply
                # shocks this top-up is intentionally disabled.
                if not bengaluru_shortage and not chennai_shortage:
                    target = spec["reorder_point"] * 2.2
                    if inventory[key] < spec["reorder_point"] * 1.15:
                        inventory[key] += max(
                            0.0,
                            target - inventory[key],
                        )

    return pd.DataFrame(rows)


def build_customer_success(
    rng: np.random.Generator,
    params: SplitParameters,
    loaded_at: dt.datetime,
) -> pd.DataFrame:
    days = list(date_range())
    rows: list[dict[str, Any]] = []
    region_ar = {r: ar1_series(rng, len(days), phi=0.65, sigma=0.05) for r in REGIONS}

    base_active = {
        "Mumbai": 4800,
        "Bengaluru": 4400,
        "Delhi": 4550,
        "Chennai": 3900,
        "Hyderabad": 3700,
    }

    for region in REGIONS:
        for i, d in enumerate(days):
            active = int(
                round(
                    base_active[region]
                    * (1.0 + 0.00045 * i)
                    * (1.0 + 0.006 * region_ar[region][i])
                )
            )

            uptime = 0.9983 + rng.normal(0.0, 0.00055)
            support_base = 31.0 * REGION_FACTOR[region] * (1.0 + 0.05 * region_ar[region][i])

            if region == "Delhi" and within(
                d, DELHI_UPTIME_DIP_DATE, dt.date(2024, 8, 31)
            ):
                uptime -= 0.0185

            support = support_base
            if region == "Delhi" and d >= DELHI_SUPPORT_SPIKE_DATE:
                # Persistent enough for the lagged relationship to be identifiable;
                # it gradually decays after September.
                if d <= dt.date(2024, 9, 10):
                    support *= params.delhi_support_multiplier
                elif d <= dt.date(2024, 9, 25):
                    decay = (dt.date(2024, 9, 25) - d).days / 15.0
                    support *= 1.0 + (params.delhi_support_multiplier - 1.0) * max(0.0, decay)

            support_tickets = max(1, int(round(support * rng.normal(1.0, 0.06))))

            churn_rate = 0.0046 + rng.normal(0.0, 0.00028)
            if region == "Delhi" and d >= DELHI_CHURN_DATE:
                if d <= dt.date(2024, 9, 15):
                    churn_rate *= params.delhi_churn_multiplier
                elif d <= dt.date(2024, 9, 30):
                    decay = (dt.date(2024, 9, 30) - d).days / 15.0
                    churn_rate *= 1.0 + (params.delhi_churn_multiplier - 1.0) * max(0.0, decay)

            churned = max(0, int(round(active * clamp(churn_rate, 0.0015, 0.02))))

            rows.append(
                {
                    "activity_date": d,
                    "region": region,
                    "active_customers": active,
                    "churned_customers": churned,
                    "support_tickets": support_tickets,
                    "platform_uptime_pct": round(float(clamp(uptime, 0.94, 1.0)), 4),
                    "source_loaded_at": loaded_at,
                }
            )

    return pd.DataFrame(rows)


def build_business_calendar(loaded_at: dt.datetime) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for region in REGIONS:
        rows.append(
            {
                "calendar_date": DIWALI_DATE,
                "region": region,
                "event_name": "Diwali",
                "event_type": "holiday",
                "expected_direction": "demand_up",
                "is_known_in_advance": 1,
                "notes": "Expected short-lived seasonal demand uplift; not a structural RCA event.",
                "source_loaded_at": loaded_at,
            }
        )

    rows.extend(
        [
            {
                "calendar_date": CHENNAI_CAMPAIGN_START,
                "region": "Chennai",
                "event_name": "Chennai year-end acquisition campaign",
                "event_type": "marketing_campaign",
                "expected_direction": "sessions_up",
                "is_known_in_advance": 1,
                "notes": "Internal campaign overlaps the severe-weather ambiguity window.",
                "source_loaded_at": loaded_at,
            },
            {
                "calendar_date": ORBIT_LAUNCH_DATE,
                "region": "Bengaluru",
                "event_name": "Orbit New product launch",
                "event_type": "product_launch",
                "expected_direction": "new_product_history_sparse",
                "is_known_in_advance": 1,
                "notes": "Used to test product-level sparse-history guardrails.",
                "source_loaded_at": loaded_at,
            },
        ]
    )
    return pd.DataFrame(rows)


def build_metric_registry(loaded_at: dt.datetime) -> pd.DataFrame:
    definitions = [
        ("revenue", "revenue = units_sold * average_selling_price", "daily_region", "Finance Analytics"),
        ("conversion_rate", "conversion_rate = orders / sessions", "daily_region", "Growth Analytics"),
        ("customer_acquisition_cost", "customer_acquisition_cost = marketing_spend / new_customers", "daily_region", "Growth Analytics"),
        ("stock_availability", "stock_availability (single metric)", "daily_region", "Operations Analytics"),
        ("churn_rate", "churn_rate = churned_customers / total_active_customers", "daily_region", "Customer Analytics"),
    ]
    return pd.DataFrame(
        [
            {
                "metric_name": name,
                "definition_version": "1.0",
                "effective_start": START_DATE,
                "effective_end": None,
                "formula_text": formula,
                "grain": grain,
                "owner_team": owner,
                "is_current": 1,
                "source_loaded_at": loaded_at,
            }
            for name, formula, grain, owner in definitions
        ]
    )


def build_source_health(loaded_at: dt.datetime) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_cadence = {
        "sales": 24,
        "marketing": 24,
        "market_context": 168,
        "customer_success": 24,
    }

    for d in date_range():
        for region in REGIONS:
            for source, cadence in source_cadence.items():
                lag = {
                    "sales": 5.5,
                    "marketing": 6.5,
                    "market_context": 20.0,
                    "customer_success": 7.0,
                }[source]
                completeness = 100.0
                status = "healthy"
                notes = None

                if (
                    source == "marketing"
                    and region == "Hyderabad"
                    and within(d, HYDERABAD_STALE_START, HYDERABAD_STALE_END)
                ):
                    lag = 78.0
                    completeness = 0.0
                    status = "stale"
                    notes = "Intentional benchmark feed outage; underlying business remains stable."

                rows.append(
                    {
                        "health_date": d,
                        "source_name": source,
                        "region": region,
                        "expected_cadence_hours": cadence,
                        "observed_lag_hours": round(lag, 2),
                        "completeness_pct": round(completeness, 2),
                        "schema_version": "1.0",
                        "status": status,
                        "notes": notes,
                        "source_loaded_at": loaded_at,
                    }
                )

    return pd.DataFrame(rows)


def build_scenario_truth(split: str, params: SplitParameters) -> dict[str, Any]:
    # Compatibility fields (kpi, changepoint_date, expected_ranks, etc.) are
    # intentionally retained for the current eval harness.
    scenarios = [
        {
            "scenario_id": "mumbai_competitor_price_pressure",
            "scenario_type": "deterministic_structural_driver",
            "kpi": "revenue",
            "region": "Mumbai",
            "intervention_date": COMPETITOR_DATE.isoformat(),
            "changepoint_date": COMPANY_PRICE_RESPONSE_DATE.isoformat(),
            "changepoint_tolerance_days": 3,
            "expected_component": "average_selling_price",
            "expected_drivers": ["competitor_price_index"],
            "expected_ranks": {"competitor_price_index": 1},
            "expected_lag_days": 2,
            "expected_direction": "down",
            "retrieval_expected": False,
        },
        {
            "scenario_id": "bengaluru_inventory_shortage",
            "scenario_type": "deterministic_supply_driver",
            "kpi": "revenue",
            "region": "Bengaluru",
            "intervention_date": INVENTORY_SHORTAGE_DATE.isoformat(),
            "changepoint_date": INVENTORY_SHORTAGE_DATE.isoformat(),
            "changepoint_tolerance_days": 3,
            "expected_component": "units_sold",
            "expected_drivers": ["stock_availability"],
            "expected_ranks": {"stock_availability": 1},
            "expected_lag_days": 0,
            "expected_direction": "down",
            "retrieval_expected": False,
        },
        {
            "scenario_id": "mumbai_heavy_rain_negative_control",
            "scenario_type": "real_external_event_no_business_effect",
            "kpi": "revenue",
            "region": "Mumbai",
            "intervention_date": MUMBAI_RAIN_DATE.isoformat(),
            "changepoint_date": None,
            "expected_component": None,
            "expected_drivers": [],
            "expected_ranks": {},
            "expected_lag_days": None,
            "expected_direction": "no_business_effect",
            "external_anchor": "Mumbai heavy rainfall",
            "retrieval_expected": False,
        },
        {
            "scenario_id": "delhi_churn_cascade",
            "scenario_type": "lagged_multidriver_cascade",
            "kpi": "churn_rate",
            "region": "Delhi",
            "intervention_date": DELHI_SUPPORT_SPIKE_DATE.isoformat(),
            "changepoint_date": DELHI_CHURN_DATE.isoformat(),
            "changepoint_tolerance_days": 3,
            "expected_component": "churned_customers",
            "expected_drivers": ["support_ticket_volume", "platform_uptime_pct"],
            "expected_ranks": {
                "support_ticket_volume": 1,
                "platform_uptime_pct": 2,
            },
            "expected_lag_days": 7,
            "secondary_expected_lag_days": 3,
            "expected_direction": "up",
            "retrieval_expected": False,
        },
        {
            "scenario_id": "hyderabad_stale_source_guardrail",
            "scenario_type": "data_quality",
            "kpi": "conversion_rate",
            "region": "Hyderabad",
            "intervention_date": HYDERABAD_STALE_START.isoformat(),
            "changepoint_date": None,
            "expected_component": None,
            "expected_drivers": [],
            "expected_ranks": {},
            "expected_lag_days": None,
            "expected_direction": "observability_failure_not_business_shift",
            "expected_behavior": "low_confidence_or_monitor",
            "expected_unhealthy_sources": ["marketing"],
            "expected_source_health_status": "stale",
            "retrieval_expected": False,
        },
        {
            "scenario_id": "diwali_seasonality_control",
            "scenario_type": "expected_calendar_effect",
            "kpi": "revenue",
            "region": "ALL",
            "intervention_date": DIWALI_DATE.isoformat(),
            "changepoint_date": None,
            "expected_component": None,
            "expected_drivers": [],
            "expected_ranks": {},
            "expected_lag_days": None,
            "expected_direction": "temporary_expected_uplift",
            "retrieval_expected": False,
        },
        {
            "scenario_id": "chennai_fengal_ambiguous_external",
            "scenario_type": "ambiguous_external_retrieval",
            "kpi": "revenue",
            "region": "Chennai",
            "intervention_date": CHENNAI_FENGAL_DATE.isoformat(),
            "changepoint_date": CHENNAI_FENGAL_DATE.isoformat(),
            "changepoint_tolerance_days": 4,
            "expected_component": "units_sold",
            # Exact deterministic ordering is intentionally NOT asserted.
            "expected_drivers": ["weather_index", "stock_availability"],
            "expected_ranks": {},
            "expected_lag_days": None,
            "expected_direction": "intentionally_ambiguous",
            "expected_ambiguity": True,
            "expected_retrieval_target": "web",
            "retrieval_expected": True,
            "external_anchor": "Chennai severe weather around 2024-11-30",
            "expected_external_entity_keywords": [
                "Fengal",
                "cyclone",
                "Chennai",
                "Tamil Nadu",
            ],
        },
        {
            "scenario_id": "orbit_new_sparse_history",
            "scenario_type": "sparse_history",
            "kpi": "revenue",
            "region": "Bengaluru",
            "product_id": "orbit_new",
            "intervention_date": ORBIT_LAUNCH_DATE.isoformat(),
            "changepoint_date": None,
            "expected_component": None,
            "expected_drivers": [],
            "expected_ranks": {},
            "expected_lag_days": None,
            "expected_direction": "sparse_history_route",
            "retrieval_expected": False,
        },
        {
            "scenario_id": "hyderabad_clean_control",
            "scenario_type": "clean_negative_control",
            "kpi": "revenue",
            "region": "Hyderabad",
            "intervention_date": None,
            "changepoint_date": None,
            "expected_component": None,
            "expected_drivers": [],
            "expected_ranks": {},
            "expected_lag_days": None,
            "expected_direction": "no_change_baseline",
            "retrieval_expected": False,
        },
    ]

    return {
        "benchmark_name": "Aletheia Business RCA Benchmark",
        "benchmark_version": "1.0",
        "split": split,
        "seed": SEEDS[split],
        "date_range": {
            "start": START_DATE.isoformat(),
            "end": END_DATE.isoformat(),
        },
        "parameters": {
            "competitor_drop": params.competitor_drop,
            "company_price_response": params.company_price_response,
            "bengaluru_replenishment_fraction": params.bengaluru_replenishment_fraction,
            "chennai_replenishment_fraction": params.chennai_replenishment_fraction,
            "chennai_weather_demand_factor": params.chennai_weather_demand_factor,
            "chennai_campaign_multiplier": params.chennai_campaign_multiplier,
            "delhi_support_multiplier": params.delhi_support_multiplier,
            "delhi_churn_multiplier": params.delhi_churn_multiplier,
        },
        "scenarios": scenarios,
    }


def build_retrieval_gold() -> dict[str, Any]:
    return {
        "scenario_id": "chennai_fengal_ambiguous_external",
        "region": "Chennai",
        "incident_date": CHENNAI_FENGAL_DATE.isoformat(),
        "gold_keywords": ["Fengal", "cyclone", "Chennai", "Tamil Nadu"],
        "gold_sources": [
            {
                "publisher": "India Meteorological Department",
                "url": "https://internal.imd.gov.in/press_release/20241130_pr_3443.pdf",
                "expected_support": "weather_external_disruption",
            },
            {
                "publisher": "Tamil Nadu Department of Information and Public Relations",
                "url": "https://dipr.tn.gov.in/",
                "expected_support": "weather_external_disruption",
            },
        ],
        "notes": (
            "Gold-source metadata is evaluation-only. URLs/keywords must never be "
            "read by src/* or app.py. Live retrieval remains independent."
        ),
    }


def validate_frames(
    sales: pd.DataFrame,
    marketing: pd.DataFrame,
    context: pd.DataFrame,
    success: pd.DataFrame,
    calendar: pd.DataFrame,
    metric_registry: pd.DataFrame,
    source_health: pd.DataFrame,
) -> None:
    # Grain uniqueness.
    assert not sales.duplicated(["sale_date", "region", "product_id"]).any()
    assert not marketing.duplicated(["activity_date", "region", "channel"]).any()
    assert not context.duplicated(["week_start", "region"]).any()
    assert not success.duplicated(["activity_date", "region"]).any()
    assert not calendar.duplicated(["calendar_date", "region", "event_name"]).any()
    assert not metric_registry.duplicated(["metric_name", "definition_version"]).any()
    assert not source_health.duplicated(["health_date", "source_name", "region"]).any()

    # Domain constraints.
    assert (sales["units_sold"] >= 0).all()
    assert (sales["inventory_units"] >= 0).all()
    assert sales["discount_pct"].between(0, 1).all()
    assert marketing[["spend", "impressions", "sessions", "orders", "new_customers"]].ge(0).all().all()
    assert (marketing["orders"] <= marketing["sessions"]).all()
    assert (marketing["new_customers"] <= marketing["orders"]).all()
    assert context["competitor_price_index"].between(0.5, 1.5).all()
    assert context["weather_index"].between(0.2, 1.5).all()
    assert success["platform_uptime_pct"].between(0, 1).all()
    assert success["churned_customers"].le(success["active_customers"]).all()

    # Intentional missing marketing feed is exactly three days x four channels.
    expected_full_marketing = len(list(date_range())) * len(REGIONS) * len(CHANNELS)
    assert expected_full_marketing - len(marketing) == 3 * len(CHANNELS)

    orbit = sales[sales["product_id"] == "orbit_new"]
    assert not orbit.empty
    assert orbit["region"].eq("Bengaluru").all()
    assert pd.to_datetime(orbit["sale_date"]).dt.date.min() == ORBIT_LAUNCH_DATE


def scenario_sanity_report(
    sales: pd.DataFrame,
    marketing: pd.DataFrame,
    context: pd.DataFrame,
    success: pd.DataFrame,
) -> dict[str, Any]:
    sales_daily = (
        sales.groupby(["sale_date", "region"], as_index=False)
        .agg(
            revenue=("net_revenue", "sum"),
            units_sold=("units_sold", "sum"),
            stock_availability=(
                "inventory_units",
                lambda s: float("nan"),
            ),
        )
    )
    # Recompute stock availability from row-level inventory/reorder point.
    stock = sales.copy()
    stock["available"] = (stock["inventory_units"] >= stock["reorder_point"]).astype(float)
    stock_daily = stock.groupby(["sale_date", "region"])["available"].mean().reset_index()
    sales_daily = sales_daily.drop(columns=["stock_availability"]).merge(
        stock_daily, on=["sale_date", "region"], how="left"
    )

    marketing_daily = (
        marketing.groupby(["activity_date", "region"], as_index=False)
        .agg(spend=("spend", "sum"), sessions=("sessions", "sum"), orders=("orders", "sum"))
    )
    marketing_daily["conversion_rate"] = (
        marketing_daily["orders"] / marketing_daily["sessions"].replace(0, np.nan)
    )

    success2 = success.copy()
    success2["churn_rate"] = success2["churned_customers"] / success2["active_customers"]

    def mean_window(df, date_col, region, col, start, end):
        dates = pd.to_datetime(df[date_col]).dt.date
        mask = (df["region"] == region) & (dates >= start) & (dates <= end)
        return float(df.loc[mask, col].mean())

    report = {
        "mumbai_revenue_pre": mean_window(
            sales_daily, "sale_date", "Mumbai", "revenue",
            dt.date(2024, 4, 7), dt.date(2024, 4, 14)
        ),
        "mumbai_revenue_post": mean_window(
            sales_daily, "sale_date", "Mumbai", "revenue",
            dt.date(2024, 4, 18), dt.date(2024, 4, 25)
        ),
        "bengaluru_stock_pre": mean_window(
            sales_daily, "sale_date", "Bengaluru", "available",
            dt.date(2024, 6, 2), dt.date(2024, 6, 9)
        ),
        "bengaluru_stock_post": mean_window(
            sales_daily, "sale_date", "Bengaluru", "available",
            dt.date(2024, 6, 11), dt.date(2024, 6, 18)
        ),
        "chennai_spend_pre": mean_window(
            marketing_daily, "activity_date", "Chennai", "spend",
            dt.date(2024, 11, 20), dt.date(2024, 11, 27)
        ),
        "chennai_spend_event": mean_window(
            marketing_daily, "activity_date", "Chennai", "spend",
            dt.date(2024, 11, 30), dt.date(2024, 12, 6)
        ),
        "chennai_stock_pre": mean_window(
            sales_daily, "sale_date", "Chennai", "available",
            dt.date(2024, 11, 20), dt.date(2024, 11, 27)
        ),
        "chennai_stock_event": mean_window(
            sales_daily, "sale_date", "Chennai", "available",
            dt.date(2024, 11, 30), dt.date(2024, 12, 6)
        ),
        "delhi_churn_pre": mean_window(
            success2, "activity_date", "Delhi", "churn_rate",
            dt.date(2024, 8, 5), dt.date(2024, 8, 11)
        ),
        "delhi_churn_post": mean_window(
            success2, "activity_date", "Delhi", "churn_rate",
            dt.date(2024, 8, 20), dt.date(2024, 8, 27)
        ),
    }
    return report


def write_split(
    split: str,
    output_root: Path,
    spec_path: Path | None,
    loaded_at: dt.datetime,
) -> dict[str, Any]:
    seed = SEEDS[split]
    rng = np.random.default_rng(seed)
    params = split_parameters(rng)

    context = build_market_context(rng, params, loaded_at)
    competitor_daily, _, _ = daily_context_maps(context)
    marketing = build_marketing(rng, params, competitor_daily, loaded_at)
    spend_map = marketing_daily_spend(marketing)
    sales = build_sales(rng, params, spend_map, loaded_at)
    success = build_customer_success(rng, params, loaded_at)
    calendar = build_business_calendar(loaded_at)
    metric_registry = build_metric_registry(loaded_at)
    source_health = build_source_health(loaded_at)

    validate_frames(
        sales, marketing, context, success, calendar, metric_registry, source_health
    )

    target = output_root / split
    target.mkdir(parents=True, exist_ok=True)

    frames = {
        "sales.csv": sales,
        "marketing.csv": marketing,
        "market_context.csv": context,
        "customer_success.csv": success,
        "business_calendar.csv": calendar,
        "metric_registry.csv": metric_registry,
        "source_health.csv": source_health,
    }
    for name, frame in frames.items():
        frame.to_csv(target / name, index=False)

    truth = build_scenario_truth(split, params)
    (target / "scenario_truth.json").write_text(
        json.dumps(truth, indent=2, default=str),
        encoding="utf-8",
    )
    (target / "retrieval_gold.json").write_text(
        json.dumps(build_retrieval_gold(), indent=2),
        encoding="utf-8",
    )

    if spec_path and spec_path.exists():
        shutil.copy2(spec_path, target / "benchmark_spec_snapshot.yaml")

    sanity = scenario_sanity_report(sales, marketing, context, success)

    manifest = {
        "benchmark_name": "Aletheia Business RCA Benchmark",
        "benchmark_version": "1.0",
        "split": split,
        "seed": seed,
        "generated_at": loaded_at.isoformat(),
        "business_date_start": START_DATE.isoformat(),
        "business_date_end": END_DATE.isoformat(),
        "row_counts": {name.removesuffix(".csv"): len(df) for name, df in frames.items()},
        "intentional_missingness": {
            "marketing": {
                "region": "Hyderabad",
                "start": HYDERABAD_STALE_START.isoformat(),
                "end": HYDERABAD_STALE_END.isoformat(),
                "missing_rows": 12,
            }
        },
        "sanity": sanity,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["dev", "holdout", "both"],
        default="both",
    )
    parser.add_argument(
        "--output-root",
        default="data/generated",
    )
    parser.add_argument(
        "--spec",
        default="benchmark_spec_v1.yaml",
    )
    parser.add_argument(
        "--ingestion-timestamp",
        default=None,
        help="ISO timestamp; defaults to generation time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    spec_path = Path(args.spec) if args.spec else None
    loaded_at = (
        dt.datetime.fromisoformat(args.ingestion_timestamp)
        if args.ingestion_timestamp
        else dt.datetime.now().replace(microsecond=0)
    )

    splits = ["dev", "holdout"] if args.split == "both" else [args.split]
    for split in splits:
        manifest = write_split(split, output_root, spec_path, loaded_at)
        print(
            f"{split}: "
            + ", ".join(f"{k}={v}" for k, v in manifest["row_counts"].items())
        )
        print("  sanity:", json.dumps(manifest["sanity"], indent=2))


if __name__ == "__main__":
    main()
