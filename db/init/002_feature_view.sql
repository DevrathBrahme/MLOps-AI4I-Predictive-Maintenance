CREATE OR REPLACE VIEW sensor_features AS
SELECT
    udi,
    product_id,
    type,
    air_temp_k,
    process_temp_k,
    rotational_speed_rpm,
    torque_nm,
    tool_wear_min,
    machine_failure, twf, hdf, pwf, osf, rnf,   -- flags kept visible for monitoring

    torque_nm * ((rotational_speed_rpm * 2 * pi())/60) AS power_w,
    (process_temp_k - air_temp_k) AS temp_diff_k,
    tool_wear_min * torque_nm AS strain_min_nm,

    CASE
        WHEN NOT machine_failure THEN 'no_failure'
        WHEN twf THEN 'TWF'
        WHEN pwf THEN 'PWF'
        WHEN osf THEN 'OSF'
        WHEN hdf THEN 'HDF'
        ELSE NULL
    END AS failure_type
FROM sensor_readings;