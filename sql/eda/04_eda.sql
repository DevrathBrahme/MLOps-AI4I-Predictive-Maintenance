SELECT
    failure_type,
    COUNT(*) AS n_rows,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM sensor_features
GROUP BY failure_type
ORDER BY n_rows DESC;

SELECT
    failure_type,
    ROUND(AVG(power_w)::numeric, 1)       AS avg_power_w,
    ROUND(AVG(temp_diff_k)::numeric, 2)   AS avg_temp_diff_k,
    ROUND(AVG(strain_min_nm)::numeric, 1) AS avg_strain_min,
    ROUND(AVG(tool_wear_min)::numeric, 0) AS avg_tool_wear_min,
    ROUND(AVG(rotational_speed_rpm)::numeric, 0)       AS avg_rotational_speed,
    ROUND(AVG(torque_nm)::numeric, 1)                   AS avg_torque
FROM sensor_features
GROUP BY failure_type
ORDER BY failure_type;

SELECT
    COUNT(*) FILTER (WHERE tool_wear_min BETWEEN 200 and 240) AS in_window,
    COUNT(*) FILTER (WHERE tool_wear_min BETWEEN 200 and 240 AND twf) AS in_window_twf
FROM sensor_features;

SELECT
    type,
    COUNT(*) FILTER (WHERE failure_type = 'no_failure') AS no_failure,
    COUNT(*) FILTER (WHERE failure_type = 'TWF') AS twf,
    COUNT(*) FILTER (WHERE failure_type = 'HDF') AS hdf,
    COUNT(*) FILTER (WHERE failure_type = 'PWF') AS pwf,
    COUNT(*) FILTER (WHERE failure_type = 'OSF') AS osf
FROM sensor_features
GROUP BY type
ORDER BY type;