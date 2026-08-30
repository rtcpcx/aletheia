-- Aletheia Business RCA Benchmark v1.0
-- Post-load verification queries.

SELECT 'sales' AS table_name, COUNT(*) AS row_count FROM raw.sales
UNION ALL SELECT 'marketing', COUNT(*) FROM raw.marketing
UNION ALL SELECT 'market_context', COUNT(*) FROM raw.market_context
UNION ALL SELECT 'customer_success', COUNT(*) FROM raw.customer_success
UNION ALL SELECT 'business_calendar', COUNT(*) FROM raw.business_calendar
UNION ALL SELECT 'metric_registry', COUNT(*) FROM raw.metric_registry
UNION ALL SELECT 'source_health', COUNT(*) FROM raw.source_health;

SELECT COUNT(*) AS mart_rows
FROM mart.daily_kpi_evidence;

-- Grain/duplicate checks: every result should be zero.
SELECT COUNT(*) AS duplicate_sales_grains
FROM (
    SELECT sale_date, region, product_id, COUNT(*) c
    FROM raw.sales
    GROUP BY sale_date, region, product_id
    HAVING COUNT(*) > 1
) x;

SELECT COUNT(*) AS duplicate_marketing_grains
FROM (
    SELECT activity_date, region, channel, COUNT(*) c
    FROM raw.marketing
    GROUP BY activity_date, region, channel
    HAVING COUNT(*) > 1
) x;

-- Benchmark scenario spot checks.
SELECT metric_date, region, revenue, units_sold, average_selling_price,
       stock_availability, competitor_price_index, weather_index, event_type
FROM mart.daily_kpi_evidence
WHERE
    (region = 'Mumbai' AND metric_date BETWEEN '2024-04-12' AND '2024-04-22')
 OR (region = 'Bengaluru' AND metric_date BETWEEN '2024-06-07' AND '2024-06-18')
 OR (region = 'Chennai' AND metric_date BETWEEN '2024-11-27' AND '2024-12-07')
ORDER BY region, metric_date;

SELECT metric_date, region, churned_customers, total_active_customers,
       support_ticket_volume, platform_uptime_pct
FROM mart.daily_kpi_evidence
WHERE region = 'Delhi'
  AND metric_date BETWEEN '2024-08-09' AND '2024-08-28'
ORDER BY metric_date;

-- Intentional Hyderabad marketing outage.
SELECT *
FROM raw.source_health
WHERE region = 'Hyderabad'
  AND source_name = 'marketing'
  AND health_date BETWEEN '2024-09-22' AND '2024-09-26'
ORDER BY health_date;

-- Orbit New must have sparse product history.
SELECT region, product_id,
       MIN(sale_date) AS first_date,
       MAX(sale_date) AS last_date,
       COUNT(DISTINCT sale_date) AS observed_days
FROM raw.sales
WHERE product_id = 'orbit_new'
GROUP BY region, product_id;
