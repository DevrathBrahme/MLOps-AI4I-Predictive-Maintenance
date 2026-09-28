\x
-- Do the physics features reproduce the failure flags?
SELECT
    COUNT(*) FILTER (WHERE temp_diff_k < 8.6 AND rotational_speed_rpm < 1380)          AS hdf_rule,
    COUNT(*) FILTER (WHERE hdf)                                                        AS hdf_flag,
    COUNT(*) FILTER (WHERE (temp_diff_k < 8.6 AND rotational_speed_rpm < 1380) <> hdf) AS hdf_disagree,

    COUNT(*) FILTER (WHERE power_w < 3500 OR power_w > 9000)                           AS pwf_rule,
    COUNT(*) FILTER (WHERE pwf)                                                        AS pwf_flag,
    COUNT(*) FILTER (WHERE (power_w < 3500 OR power_w > 9000) <> pwf)                  AS pwf_disagree,

    COUNT(*) FILTER (WHERE strain_min_nm > CASE type WHEN 'L' THEN 11000 WHEN 'M' THEN 12000 WHEN 'H' THEN 13000 END)          AS osf_rule,
    COUNT(*) FILTER (WHERE osf)                                                                                                AS osf_flag,
    COUNT(*) FILTER (WHERE (strain_min_nm > CASE type WHEN 'L' THEN 11000 WHEN 'M' THEN 12000 WHEN 'H' THEN 13000 END) <> osf) AS osf_disagree

FROM sensor_features;