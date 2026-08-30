-- Aletheia — sql/03_analysis_schema.sql
-- Analysis-stage tables. Written only by src/*.py pipeline stages, never by
-- raw ingestion. analysis.retrieved_context is the ONLY table LLM retrieval
-- may write to.

CREATE DATABASE IF NOT EXISTS analysis;

CREATE TABLE analysis.detected_changepoints (
    kpi VARCHAR(64) NOT NULL,
    region VARCHAR(64) NOT NULL,
    changepoint_date DATE NOT NULL,
    segment_mean_before DECIMAL(18,6),
    segment_mean_after DECIMAL(18,6),
    shift_magnitude DECIMAL(18,6),
    method VARCHAR(32) NOT NULL,
    PRIMARY KEY (kpi, region, changepoint_date)
);

CREATE TABLE analysis.pvm_decomposition (
    kpi VARCHAR(64) NOT NULL,
    region VARCHAR(64) NOT NULL,
    window_start DATE NOT NULL,

    decomposition_type VARCHAR(32) NOT NULL,

    effect_a DECIMAL(18,8),
    effect_b DECIMAL(18,8),
    interaction_effect DECIMAL(18,8),
    residual DECIMAL(18,8),
    total_change DECIMAL(18,8),

    is_volatile BOOLEAN NOT NULL DEFAULT FALSE,
    narrative_mode VARCHAR(32) NOT NULL DEFAULT 'standard',

    PRIMARY KEY (kpi, region, window_start)
);

CREATE TABLE analysis.driver_evidence (
    kpi VARCHAR(64) NOT NULL,
    region VARCHAR(64) NOT NULL,
    window_start DATE NOT NULL,

    driver_name VARCHAR(96) NOT NULL,
    explains_component VARCHAR(96) NOT NULL,
    evidence_mode VARCHAR(32) NOT NULL,
    model_status VARCHAR(32) NOT NULL,

    best_lag_days INT NOT NULL,

    baseline_value DECIMAL(18,6),
    incident_value DECIMAL(18,6),

    driver_zscore DECIMAL(12,6),
    structural_break_score DECIMAL(12,8),

    historical_coefficient DECIMAL(18,8),
    holdout_correlation DECIMAL(12,8),

    p_value DECIMAL(12,10),
    is_significant BOOLEAN NOT NULL,

    coefficient_stability DECIMAL(12,8),
    evidence_score DECIMAL(18,8),
    normalized_score DECIMAL(12,8),
    softmax_probability DECIMAL(12,8),

    PRIMARY KEY (
        kpi,
        region,
        window_start,
        driver_name,
        explains_component
    )
);

CREATE TABLE analysis.retrieved_context (
    retrieval_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    kpi VARCHAR(64) NOT NULL,
    region VARCHAR(64) NOT NULL,
    window_start DATE NOT NULL,

    hypothesis VARCHAR(256) NOT NULL,
    retrieval_query TEXT NOT NULL,

    source_title VARCHAR(512),
    source_url TEXT,
    retrieved_text TEXT,

    retrieval_support DECIMAL(10,6) NOT NULL,
    retrieval_confidence DECIMAL(10,6) NOT NULL,

    retrieved_at DATETIME NOT NULL,

    INDEX idx_retrieved_context_case (
        kpi,
        region,
        window_start
    )
);

CREATE TABLE analysis.orchestrator_updates (
    kpi VARCHAR(64) NOT NULL,
    region VARCHAR(64) NOT NULL,
    window_start DATE NOT NULL,

    driver_name VARCHAR(96) NOT NULL,

    probability_before DECIMAL(12,8),
    retrieval_query TEXT,

    retrieval_support DECIMAL(10,6),
    probability_after DECIMAL(12,8),

    updated_at DATETIME NOT NULL,

    INDEX idx_orchestrator_case (
        kpi,
        region,
        window_start
    )
);

CREATE TABLE analysis.evidence_bundle (
    kpi VARCHAR(64) NOT NULL,
    region VARCHAR(64) NOT NULL,
    window_start DATE NOT NULL,

    bundle_json JSON NOT NULL,
    generated_at DATETIME NOT NULL,

    PRIMARY KEY (kpi, region, window_start)
);

CREATE TABLE analysis.decision_packets (
    kpi VARCHAR(64) NOT NULL,
    region VARCHAR(64) NOT NULL,
    window_start DATE NOT NULL,

    percent_change DECIMAL(12,6),
    confidence_level VARCHAR(16) NOT NULL,

    recommended_action TEXT,
    freshness_caveat TEXT,

    PRIMARY KEY (kpi, region, window_start)
);

CREATE TABLE analysis.clarification_requests (
    request_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    kpi VARCHAR(64) NOT NULL,
    region VARCHAR(64) NOT NULL,
    window_start DATE NOT NULL,

    reason TEXT NOT NULL,
    requested_clarification TEXT NOT NULL,

    created_at DATETIME NOT NULL,

    INDEX idx_clarification_case (
        kpi,
        region,
        window_start
    )
);
