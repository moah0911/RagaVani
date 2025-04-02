#!/bin/bash
# Script to build and push the RagaVani Docker image to Snowflake

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <snowflake_account_identifier> <username>"
    exit 1
fi

ACCOUNT_IDENTIFIER=$1
USERNAME=$2

echo "Building Docker image..."
docker build -t ragavani:latest .

echo "Tagging image for Snowflake registry..."
docker tag ragavani:latest ${ACCOUNT_IDENTIFIER}.registry.snowflakecomputing.com/ragavani_db/app/ragavani_repo/ragavani:latest

echo "Logging in to Snowflake registry..."
docker login ${ACCOUNT_IDENTIFIER}.registry.snowflakecomputing.com -u ${USERNAME}

echo "Pushing image to Snowflake registry..."
docker push ${ACCOUNT_IDENTIFIER}.registry.snowflakecomputing.com/ragavani_db/app/ragavani_repo/ragavani:latest

echo "Image successfully pushed to Snowflake registry."
echo "You can now deploy the service using the deploy_to_snowflake.sql script."