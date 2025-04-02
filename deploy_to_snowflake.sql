-- Snowflake deployment script for RagaVani
-- This script sets up the necessary Snowflake objects for deploying RagaVani

-- Create a warehouse for the application
CREATE WAREHOUSE IF NOT EXISTS RAGAVANI_WH
  WITH WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 300
  AUTO_RESUME = TRUE;

-- Create a database for the application
CREATE DATABASE IF NOT EXISTS RAGAVANI_DB;

-- Create a schema for the application
CREATE SCHEMA IF NOT EXISTS RAGAVANI_DB.APP;

-- Create a role for the application
CREATE ROLE IF NOT EXISTS RAGAVANI_ROLE;

-- Grant privileges to the role
GRANT USAGE ON WAREHOUSE RAGAVANI_WH TO ROLE RAGAVANI_ROLE;
GRANT USAGE ON DATABASE RAGAVANI_DB TO ROLE RAGAVANI_ROLE;
GRANT USAGE ON SCHEMA RAGAVANI_DB.APP TO ROLE RAGAVANI_ROLE;
GRANT CREATE TABLE ON SCHEMA RAGAVANI_DB.APP TO ROLE RAGAVANI_ROLE;
GRANT CREATE VIEW ON SCHEMA RAGAVANI_DB.APP TO ROLE RAGAVANI_ROLE;

-- Create a compute pool for Snowpark Container Services
CREATE COMPUTE POOL IF NOT EXISTS RAGAVANI_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = 'CPU_X64_STANDARD'
  AUTO_RESUME = TRUE;

-- Create an image repository for the Docker image
CREATE IMAGE REPOSITORY IF NOT EXISTS RAGAVANI_DB.APP.RAGAVANI_REPO;

-- Grant privileges on the image repository
GRANT USAGE ON IMAGE REPOSITORY RAGAVANI_DB.APP.RAGAVANI_REPO TO ROLE RAGAVANI_ROLE;
GRANT READ ON IMAGE REPOSITORY RAGAVANI_DB.APP.RAGAVANI_REPO TO ROLE RAGAVANI_ROLE;
GRANT WRITE ON IMAGE REPOSITORY RAGAVANI_DB.APP.RAGAVANI_REPO TO ROLE RAGAVANI_ROLE;

-- Create a service specification
CREATE SERVICE SPECIFICATION IF NOT EXISTS RAGAVANI_DB.APP.RAGAVANI_SPEC
  FROM '/snowflake.yml';

-- Create a service using the specification
CREATE SERVICE IF NOT EXISTS RAGAVANI_SERVICE
  IN COMPUTE POOL RAGAVANI_POOL
  FROM SPECIFICATION RAGAVANI_DB.APP.RAGAVANI_SPEC
  COMPUTE_ROLE = RAGAVANI_ROLE
  EXTERNAL_ACCESS_INTEGRATIONS = ();

-- Grant privileges on the service
GRANT USAGE ON SERVICE RAGAVANI_SERVICE TO ROLE RAGAVANI_ROLE;

-- Output the service URL
SELECT SYSTEM$GET_SERVICE_STATUS('RAGAVANI_SERVICE');