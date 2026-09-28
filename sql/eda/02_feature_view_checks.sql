-- 1. Label counts
SELECT failure_type, COUNT(*) AS n_rows
FROM sensor_features
GROUP BY failure_type
ORDER BY n_rows DESC;

-- 2. Spot-check one row
SELECT
    udi, rotational_speed_rpm, torque_nm, air_temp_k, process_temp_k, tool_wear_min,
    ROUND(power_w::numeric, 1)       AS power_w,
    ROUND(temp_diff_k::numeric, 2)   AS temp_diff_k,
    ROUND(strain_min_nm::numeric, 1) AS strain_min_nm
FROM sensor_features
WHERE udi = 1;