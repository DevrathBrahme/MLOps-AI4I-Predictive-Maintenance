CREATE TABLE predictions_log (
    request_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_name           TEXT NOT NULL,
    model_version        INTEGER NOT NULL CHECK (model_version > 0),
    type                 TEXT NOT NULL CHECK (type IN ('L', 'M', 'H')),
    air_temp_k           DOUBLE PRECISION NOT NULL CHECK (air_temp_k > 0),
    process_temp_k       DOUBLE PRECISION NOT NULL CHECK (process_temp_k > 0),
    rotational_speed_rpm INTEGER NOT NULL CHECK (rotational_speed_rpm > 0),
    torque_nm            DOUBLE PRECISION NOT NULL CHECK (torque_nm >= 0),
    tool_wear_min        INTEGER NOT NULL CHECK (tool_wear_min >= 0),
    power_w              DOUBLE PRECISION NOT NULL,
    temp_diff_k          DOUBLE PRECISION NOT NULL,
    strain_min_nm        DOUBLE PRECISION NOT NULL,
    p_no_failure         DOUBLE PRECISION NOT NULL CHECK (p_no_failure BETWEEN 0 AND 1),
    p_twf                DOUBLE PRECISION NOT NULL CHECK (p_twf BETWEEN 0 AND 1),
    p_hdf                DOUBLE PRECISION NOT NULL CHECK (p_hdf BETWEEN 0 AND 1),
    p_pwf                DOUBLE PRECISION NOT NULL CHECK (p_pwf BETWEEN 0 AND 1),
    p_osf                DOUBLE PRECISION NOT NULL CHECK (p_osf BETWEEN 0 AND 1),
    predicted_mode       TEXT NOT NULL
        CHECK (predicted_mode IN ('no_failure', 'TWF', 'HDF', 'PWF', 'OSF')),
    runner_up_mode       TEXT NOT NULL
        CHECK (runner_up_mode IN ('no_failure', 'TWF', 'HDF', 'PWF', 'OSF')),
    confidence_score     DOUBLE PRECISION NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    elevated_threshold   DOUBLE PRECISION NOT NULL CHECK (elevated_threshold BETWEEN 0 AND 1),
    risk_tier            TEXT NOT NULL CHECK (risk_tier IN ('low', 'elevated', 'high')),
    CONSTRAINT runner_up_differs CHECK (runner_up_mode <> predicted_mode),
    CONSTRAINT tier_matches_rule CHECK (
        risk_tier = CASE
            WHEN predicted_mode <> 'no_failure' THEN 'high'
            WHEN confidence_score > elevated_threshold THEN 'elevated'
            ELSE 'low'
        END
    )
);

CREATE INDEX predictions_log_created_at_idx ON predictions_log (created_at);