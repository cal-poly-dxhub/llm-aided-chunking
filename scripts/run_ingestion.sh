#!/bin/bash

# Configuration
LAMBDA_FUNCTION_NAME="your-lambda-function-name"
RUN_LOCAL=true  # Set to false to run in cloud

# Event configuration
EVENT='{
  "bucket_name": "wisconsin-chatbot-sources-west-2",
  "s3_key": "sources/pb056-board-review-guide/pb056-board-review-guide.pdf",
  "document_url": "n/a",
  "debug_mode": true,
  "primary_output_bucket": "chunks-output-bucket",
  "secondary_output_bucket": "logs-output-bucket"
}'

# Local event (uncomment to use)
# EVENT='{
#   "bucket_name": "wisconsin-chatbot-sources",
#   "s3_key": "sources/wi-statute-ch75/wi-statute-ch75.pdf",
#   "document_url": "n/a",
#   "debug_mode": true
# }'

if [ "$RUN_LOCAL" = true ]; then
    echo "Running locally..."
    echo "$EVENT" | uv run python -c "
import json
import sys
sys.path.append('lambda/ingestion')
sys.path.append('lambda/layers/textract')
sys.path.append('lambda/layers/shared')
from handler import lambda_handler

event = json.loads(sys.stdin.read())
result = lambda_handler(event, None)
print(json.dumps(result, indent=2))
" 2>&1
else
    echo "Running in cloud..."
    aws lambda invoke \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --cli-binary-format raw-in-base64-out \
        --payload "$EVENT" \
        response.json
    cat response.json
    rm response.json
fi
