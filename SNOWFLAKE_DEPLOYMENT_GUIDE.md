# RagaVani Deployment Guide for Snowflake

This guide provides step-by-step instructions for deploying RagaVani on Snowflake using Snowpark Container Services.

## Prerequisites

- Snowflake account with access to Snowpark Container Services
- Snowflake account with ACCOUNTADMIN role or equivalent privileges
- Docker installed on your local machine
- Snowflake CLI installed on your local machine

## Deployment Steps

### 1. Set Up Snowflake Environment

1. Connect to your Snowflake account using the Snowflake CLI:

```bash
snowsql -a <account_identifier> -u <username>
```

2. Run the deployment script to set up the necessary Snowflake objects:

```bash
snowsql -a <account_identifier> -u <username> -f deploy_to_snowflake.sql
```

### 2. Build and Push Docker Image

1. Build the Docker image locally:

```bash
docker build -t ragavani:latest .
```

2. Tag the image for the Snowflake repository:

```bash
docker tag ragavani:latest <account_identifier>.registry.snowflakecomputing.com/ragavani_db/app/ragavani_repo/ragavani:latest
```

3. Log in to the Snowflake registry:

```bash
docker login <account_identifier>.registry.snowflakecomputing.com -u <username>
```

4. Push the image to the Snowflake registry:

```bash
docker push <account_identifier>.registry.snowflakecomputing.com/ragavani_db/app/ragavani_repo/ragavani:latest
```

### 3. Deploy the Service

1. Connect to Snowflake using SnowSQL:

```bash
snowsql -a <account_identifier> -u <username>
```

2. Create the service using the specification:

```sql
USE ROLE RAGAVANI_ROLE;
USE DATABASE RAGAVANI_DB;
USE SCHEMA APP;

CREATE OR REPLACE SERVICE RAGAVANI_SERVICE
  IN COMPUTE POOL RAGAVANI_POOL
  FROM SPECIFICATION RAGAVANI_SPEC
  COMPUTE_ROLE = RAGAVANI_ROLE
  EXTERNAL_ACCESS_INTEGRATIONS = ();
```

3. Check the service status:

```sql
SELECT SYSTEM$GET_SERVICE_STATUS('RAGAVANI_SERVICE');
```

4. Get the service URL:

```sql
SELECT SYSTEM$GET_SERVICE_ENDPOINT('RAGAVANI_SERVICE', 'ragavani');
```

### 4. Access the Application

1. Open the service URL in a web browser to access the RagaVani application.

2. The application should be running and accessible through the provided URL.

## Troubleshooting

### Service Not Starting

If the service fails to start, check the service logs:

```sql
SELECT SYSTEM$GET_SERVICE_LOGS('RAGAVANI_SERVICE', 0, 'container');
```

### Image Pull Issues

If there are issues pulling the image, verify that the image repository is correctly set up and that the image has been pushed:

```sql
SHOW IMAGE REPOSITORIES IN DATABASE RAGAVANI_DB;
LIST @RAGAVANI_DB.APP.RAGAVANI_REPO;
```

### Permission Issues

If there are permission issues, verify that the RAGAVANI_ROLE has the necessary privileges:

```sql
SHOW GRANTS TO ROLE RAGAVANI_ROLE;
```

## Data Persistence

The application is configured to store data in the `/data` directory, which is mounted as a volume in the container. This ensures that user data persists between service restarts.

## Scaling

To scale the application, you can adjust the compute pool settings:

```sql
ALTER COMPUTE POOL RAGAVANI_POOL
  SET MIN_NODES = 2
  MAX_NODES = 4;
```

## Monitoring

Monitor the service using Snowflake's built-in monitoring capabilities:

```sql
-- Check service status
SELECT SYSTEM$GET_SERVICE_STATUS('RAGAVANI_SERVICE');

-- Check service metrics
SELECT * FROM TABLE(INFORMATION_SCHEMA.SERVICE_METRICS('RAGAVANI_SERVICE'));
```

## Updating the Application

To update the application:

1. Build and push a new Docker image with the updated code.
2. Restart the service to use the new image:

```sql
ALTER SERVICE RAGAVANI_SERVICE REFRESH;
```