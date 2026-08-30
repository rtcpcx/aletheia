-- Aletheia — governed feedback integration
-- Idempotent migration for app.user_feedback. No analytical tables are changed.

CREATE DATABASE IF NOT EXISTS app;

-- kpi
SET @ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE app.user_feedback ADD COLUMN kpi VARCHAR(128) NULL AFTER region',
        'SELECT 1'
    )
    FROM information_schema.columns
    WHERE table_schema = 'app' AND table_name = 'user_feedback' AND column_name = 'kpi'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- window_start
SET @ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE app.user_feedback ADD COLUMN window_start DATE NULL AFTER kpi',
        'SELECT 1'
    )
    FROM information_schema.columns
    WHERE table_schema = 'app' AND table_name = 'user_feedback' AND column_name = 'window_start'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- primary_driver
SET @ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE app.user_feedback ADD COLUMN primary_driver VARCHAR(128) NULL AFTER window_start',
        'SELECT 1'
    )
    FROM information_schema.columns
    WHERE table_schema = 'app' AND table_name = 'user_feedback' AND column_name = 'primary_driver'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- action_level
SET @ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE app.user_feedback ADD COLUMN action_level VARCHAR(64) NULL AFTER primary_driver',
        'SELECT 1'
    )
    FROM information_schema.columns
    WHERE table_schema = 'app' AND table_name = 'user_feedback' AND column_name = 'action_level'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- is_demo
SET @ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE app.user_feedback ADD COLUMN is_demo TINYINT(1) NOT NULL DEFAULT 0 AFTER comment_text',
        'SELECT 1'
    )
    FROM information_schema.columns
    WHERE table_schema = 'app' AND table_name = 'user_feedback' AND column_name = 'is_demo'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- Pattern lookup index (idempotent).
SET @ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'CREATE INDEX idx_feedback_pattern_created ON app.user_feedback (kpi, primary_driver, action_level, created_at)',
        'SELECT 1'
    )
    FROM information_schema.statistics
    WHERE table_schema = 'app'
      AND table_name = 'user_feedback'
      AND index_name = 'idx_feedback_pattern_created'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;
