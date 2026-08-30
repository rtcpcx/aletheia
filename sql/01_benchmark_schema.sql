-- Aletheia Business RCA Benchmark v1.0
-- Full raw/mart schema for a clean benchmark rebuild.
--
-- Intended usage after dropping old raw/mart databases:
--   Get-Content sql\01_benchmark_schema.sql | mysql -u root -p
--
-- Core business facts preserve the columns expected by the existing Aletheia
-- pipeline. Primary keys enforce the declared benchmark grains.

CREATE DATABASE IF NOT EXISTS raw;
CREATE DATABASE IF NOT EXISTS mart;

CREATE TABLE raw.sales (
    sale_date DATE NOT NULL,
    region VARCHAR(64) NOT NULL,
    product_id VARCHAR(64) NOT NULL,

    units_sold INT NOT NULL,
    list_price DECIMAL(12,2) NOT NULL,
    unit_cost DECIMAL(12,2) NOT NULL,
    discount_pct DECIMAL(5,4) NOT NULL,
    net_revenue DECIMAL(14,2) NOT NULL,

    inventory_units INT NOT NULL,
    reorder_point INT NOT NULL,

    source_loaded_at DATETIME NOT NULL,

    PRIMARY KEY (sale_date, region, product_id),
    INDEX idx_sales_region_date (region, sale_date),
    INDEX idx_sales_product_region_date (product_id, region, sale_date)
);

CREATE TABLE raw.marketing (
    activity_date DATE NOT NULL,
    region VARCHAR(64) NOT NULL,
    channel VARCHAR(64) NOT NULL,

    spend DECIMAL(14,2) NOT NULL,
    impressions INT NOT NULL,
    sessions INT NOT NULL,
    orders INT NOT NULL,
    new_customers INT NOT NULL,

    source_loaded_at DATETIME NOT NULL,

    PRIMARY KEY (activity_date, region, channel),
    INDEX idx_marketing_region_date (region, activity_date)
);

CREATE TABLE raw.market_context (
    week_start DATE NOT NULL,
    region VARCHAR(64) NOT NULL,

    competitor_price_index DECIMAL(8,4),
    weather_index DECIMAL(8,4),
    event_type VARCHAR(128),

    source_type VARCHAR(32) NOT NULL DEFAULT 'scheduled_feed',
    source_loaded_at DATETIME NOT NULL,

    PRIMARY KEY (week_start, region),
    INDEX idx_context_region_week (region, week_start),
    INDEX idx_context_event (event_type),
    INDEX idx_context_source_type (source_type)
);

CREATE TABLE raw.customer_success (
    activity_date DATE NOT NULL,
    region VARCHAR(64) NOT NULL,

    active_customers INT NOT NULL,
    churned_customers INT NOT NULL,
    support_tickets INT NOT NULL,
    platform_uptime_pct DECIMAL(6,4) NOT NULL,

    source_loaded_at DATETIME NOT NULL,

    PRIMARY KEY (activity_date, region),
    INDEX idx_success_region_date (region, activity_date)
);

-- Known-in-advance business context. This is deterministic internal metadata,
-- not web retrieval.
CREATE TABLE raw.business_calendar (
    calendar_date DATE NOT NULL,
    region VARCHAR(64) NOT NULL,
    event_name VARCHAR(160) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    expected_direction VARCHAR(64),
    is_known_in_advance BOOLEAN NOT NULL DEFAULT TRUE,
    notes TEXT,
    source_loaded_at DATETIME NOT NULL,

    PRIMARY KEY (calendar_date, region, event_name),
    INDEX idx_calendar_region_date (region, calendar_date),
    INDEX idx_calendar_type_date (event_type, calendar_date)
);

-- Versioned KPI definitions. This lets the benchmark audit definition drift
-- separately from causal/business movement.
CREATE TABLE raw.metric_registry (
    metric_name VARCHAR(96) NOT NULL,
    definition_version VARCHAR(32) NOT NULL,
    effective_start DATE NOT NULL,
    effective_end DATE,
    formula_text TEXT NOT NULL,
    grain VARCHAR(96) NOT NULL,
    owner_team VARCHAR(128),
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    source_loaded_at DATETIME NOT NULL,

    PRIMARY KEY (metric_name, definition_version),
    INDEX idx_metric_current (metric_name, is_current),
    INDEX idx_metric_effective (effective_start, effective_end)
);

-- Explicit source-health facts. A missing row in raw.marketing should not be
-- interpreted as zero marketing activity; this table records the observability
-- problem independently.
CREATE TABLE raw.source_health (
    health_date DATE NOT NULL,
    source_name VARCHAR(64) NOT NULL,
    region VARCHAR(64) NOT NULL,

    expected_cadence_hours INT NOT NULL,
    observed_lag_hours DECIMAL(10,2) NOT NULL,
    completeness_pct DECIMAL(6,2) NOT NULL,
    schema_version VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    notes TEXT,

    source_loaded_at DATETIME NOT NULL,

    PRIMARY KEY (health_date, source_name, region),
    INDEX idx_health_source_region_date (source_name, region, health_date),
    INDEX idx_health_status_date (status, health_date)
);
