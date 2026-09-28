WITH flagged AS (
    SELECT
        machine_failure,
        (twf::int + hdf::int + pwf::int + osf::int + rnf::int) AS n_modes          -- add the five mode flags, each cast to int
    FROM sensor_readings
)
SELECT machine_failure, n_modes, COUNT(*) AS n_rows
FROM flagged
GROUP BY machine_failure, n_modes
ORDER BY machine_failure, n_modes;

-- 2. Multi-mode combinations
WITH flagged AS (
    SELECT
        machine_failure,
        twf,
        hdf,
        pwf,
        osf,
        rnf,
        (twf::int + hdf::int + pwf::int + osf::int + rnf::int) AS n_modes
    FROM sensor_readings
)
SELECT twf, hdf, pwf, osf, rnf, COUNT(*) AS n_rows
FROM flagged
WHERE n_modes > 1                        -- only rows with more than one mode
GROUP BY twf, hdf, pwf, osf, rnf
ORDER BY n_rows DESC;                    -- most frequent combination first

-- 3. Modes flagged without a failure
SELECT
    COUNT(*) FILTER (WHERE twf AND NOT machine_failure) AS twf_without_failure,
    COUNT(*) FILTER (WHERE hdf AND NOT machine_failure) AS hdf_without_failure,
    COUNT(*) FILTER (WHERE pwf AND NOT machine_failure) AS pwf_without_failure,
    COUNT(*) FILTER (WHERE osf AND NOT machine_failure) AS osf_without_failure,
    COUNT(*) FILTER (WHERE rnf AND NOT machine_failure) AS rnf_without_failure
FROM sensor_readings;