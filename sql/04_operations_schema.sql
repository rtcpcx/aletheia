-- Aletheia — sql/04_operations_schema.sql
-- App-facing operational tables: user feedback and runtime telemetry.

CREATE DATABASE IF NOT EXISTS app;

CREATE TABLE app.user_feedback (
    feedback_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    persona VARCHAR(64),
    region VARCHAR(64),

    disposition VARCHAR(32),
    comment_text TEXT,

    INDEX idx_feedback_region_created (
        region,
        created_at
    )
);

CREATE TABLE app.runtime_telemetry (
    telemetry_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    operation_name VARCHAR(128),
    latency_ms INT,

    llm_calls INT DEFAULT 0,
    estimated_cost_usd DECIMAL(10,6) DEFAULT 0,

    status VARCHAR(32),

    INDEX idx_telemetry_operation_created (
        operation_name,
        created_at
    )
);
