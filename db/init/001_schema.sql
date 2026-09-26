-- This file contains the SQL statements to create the database schema for the AI4I Predictive Maintenance project.
-- The table sensor readings includes 14 columns with udi as the primary key.
-- I have used double precision for the air_temp_k, process_temp_k, and torque_nm columns as 
-- these are physical measurements, and a double's ~15 significant digits far exceed the precision of any sensor.
-- twf, hdf, pwf, osf, and rnf are boolean columns that indicate whether a specific type of failure has occurred.
CREATE TABLE if NOT EXISTS sensor_readings (
    udi INTEGER PRIMARY KEY,
    product_id VARCHAR(6) NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('L', 'M', 'H')),
    air_temp_k DOUBLE PRECISION NOT NULL,
    process_temp_k DOUBLE PRECISION NOT NULL,
    rotational_speed_rpm INTEGER NOT NULL,
    torque_nm DOUBLE PRECISION NOT NULL,
    tool_wear_min INTEGER NOT NULL,
    machine_failure BOOLEAN NOT NULL,
    twf BOOLEAN NOT NULL,
    hdf BOOLEAN NOT NULL,
    pwf BOOLEAN NOT NULL,
    osf BOOLEAN NOT NULL,
    rnf BOOLEAN NOT NULL
);