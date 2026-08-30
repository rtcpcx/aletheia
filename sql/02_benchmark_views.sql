-- Aletheia Business RCA Benchmark v1.0
-- Deterministic KPI mart + governance helper views.

CREATE DATABASE IF NOT EXISTS mart;

CREATE OR REPLACE VIEW mart.daily_kpi_evidence AS

WITH region_dates AS (
    SELECT sale_date AS metric_date, region FROM raw.sales
    UNION DISTINCT
    SELECT activity_date, region FROM raw.marketing
    UNION DISTINCT
    SELECT activity_date, region FROM raw.customer_success
),

sales_daily AS (
    SELECT
        sale_date AS metric_date,
        region,

        SUM(net_revenue) AS revenue,
        SUM(units_sold) AS units_sold,

        SUM(net_revenue)
            / NULLIF(SUM(units_sold), 0)
            AS average_selling_price,

        AVG(discount_pct) AS average_discount_pct,

        AVG(
            CASE
                WHEN inventory_units >= reorder_point THEN 1.0
                ELSE 0.0
            END
        ) AS stock_availability,

        (
            SUM(net_revenue) - SUM(units_sold * unit_cost)
        ) / NULLIF(SUM(net_revenue), 0) AS margin_pct,

        MAX(source_loaded_at) AS sales_freshness

    FROM raw.sales
    GROUP BY sale_date, region
),

marketing_daily AS (
    SELECT
        activity_date AS metric_date,
        region,

        SUM(spend) AS marketing_spend,
        SUM(sessions) AS sessions,
        SUM(orders) AS orders,
        SUM(new_customers) AS new_customers,

        SUM(orders)
            / NULLIF(SUM(sessions), 0)
            AS conversion_rate,

        SUM(spend)
            / NULLIF(SUM(new_customers), 0)
            AS customer_acquisition_cost,

        MAX(source_loaded_at) AS marketing_freshness

    FROM raw.marketing
    GROUP BY activity_date, region
),

success_daily AS (
    SELECT
        activity_date AS metric_date,
        region,

        active_customers AS total_active_customers,
        churned_customers,
        support_tickets AS support_ticket_volume,
        platform_uptime_pct,

        source_loaded_at AS success_freshness

    FROM raw.customer_success
)

SELECT
    d.metric_date,
    d.region,

    s.revenue,
    s.units_sold,
    s.average_selling_price,
    s.average_discount_pct,
    s.stock_availability,
    s.margin_pct,

    m.marketing_spend,
    m.sessions,
    m.orders,
    m.new_customers,
    m.conversion_rate,
    m.customer_acquisition_cost,

    c.competitor_price_index,
    c.weather_index,
    c.event_type,
    c.source_type AS context_source_type,

    x.total_active_customers,
    x.churned_customers,
    x.support_ticket_volume,
    x.platform_uptime_pct,

    s.sales_freshness,
    m.marketing_freshness,
    c.source_loaded_at AS context_freshness,
    x.success_freshness

FROM region_dates d

LEFT JOIN sales_daily s
    ON s.metric_date = d.metric_date
    AND s.region = d.region

LEFT JOIN marketing_daily m
    ON m.metric_date = d.metric_date
    AND m.region = d.region

LEFT JOIN raw.market_context c
    ON c.region = d.region
    AND c.week_start = DATE_SUB(
        d.metric_date,
        INTERVAL WEEKDAY(d.metric_date) DAY
    )

LEFT JOIN success_daily x
    ON x.metric_date = d.metric_date
    AND x.region = d.region;


CREATE OR REPLACE VIEW mart.source_freshness AS
SELECT
    'sales' AS source_name,
    MAX(source_loaded_at) AS latest_loaded_at,
    COUNT(*) AS row_count
FROM raw.sales

UNION ALL

SELECT
    'marketing' AS source_name,
    MAX(source_loaded_at) AS latest_loaded_at,
    COUNT(*) AS row_count
FROM raw.marketing

UNION ALL

SELECT
    'market_context' AS source_name,
    MAX(source_loaded_at) AS latest_loaded_at,
    COUNT(*) AS row_count
FROM raw.market_context

UNION ALL

SELECT
    'customer_success' AS source_name,
    MAX(source_loaded_at) AS latest_loaded_at,
    COUNT(*) AS row_count
FROM raw.customer_success;


CREATE OR REPLACE VIEW mart.source_health_latest AS
SELECT h.*
FROM raw.source_health h
JOIN (
    SELECT source_name, region, MAX(health_date) AS max_health_date
    FROM raw.source_health
    GROUP BY source_name, region
) latest
    ON latest.source_name = h.source_name
    AND latest.region = h.region
    AND latest.max_health_date = h.health_date;


CREATE OR REPLACE VIEW mart.current_metric_registry AS
SELECT
    metric_name,
    definition_version,
    effective_start,
    effective_end,
    formula_text,
    grain,
    owner_team
FROM raw.metric_registry
WHERE is_current = TRUE;


CREATE OR REPLACE VIEW mart.business_calendar_daily AS
SELECT
    calendar_date AS metric_date,
    region,
    event_name,
    event_type,
    expected_direction,
    is_known_in_advance,
    notes
FROM raw.business_calendar;
