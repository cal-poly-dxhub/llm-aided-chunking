import json
import os
import traceback
from typing import Dict, Any, Optional
import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from aws_lambda_powertools.utilities.parser import parse
from aws_lambda_powertools.utilities.parser.models import SqsModel
from tenacity import retry, stop_after_delay, wait_exponential, before_sleep_log
import logging
import instructor
from pydantic import BaseModel

from models import QueueMessage, JobConfig, Chunk
from prompt import CHUNK_CLASSIFICATION_PROMPT

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.client("dynamodb")
s3 = boto3.client("s3")
sqs = boto3.client("sqs")
brt = boto3.client("bedrock-runtime")

client = instructor.from_bedrock(brt, mode=instructor.Mode.BEDROCK_TOOLS)

deserializer = TypeDeserializer()
serializer = TypeSerializer()

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
REJECTED_BUCKET_NAME = os.environ["REJECTED_BUCKET_NAME"]
INGESTED_BUCKET_NAME = os.environ["INGESTED_BUCKET_NAME"]
DLQ_URL = os.environ["DLQ_URL"]


class ChunkClassification(BaseModel):
    accept_chunk: bool
    reason: Optional[str] = None


class MaxWaitTimeReached(Exception):
    pass


def before_sleep_hook(retry_state):
    wait_time = retry_state.next_action.sleep
    if wait_time > 30:
        raise MaxWaitTimeReached("Maximum wait time exceeded")


def raise_abort_flag(job_id: str):
    dynamodb.update_item(
        TableName=JOB_TABLE_NAME,
        Key={"job_id": {"S": job_id}},
        UpdateExpression="SET abort = :abort",
        ExpressionAttributeValues={":abort": {"BOOL": True}},
    )


def accept_chunk(chunk: Chunk):
    key = f"{chunk.metadata.doc_id}/{chunk.chunk_id}.json"
    body = {"text": chunk.text, "metadata": chunk.metadata.model_dump()}

    s3.put_object(
        Bucket=INGESTED_BUCKET_NAME,
        Key=key,
        Body=json.dumps(body),
        ContentType="application/json",
    )


def reject_chunk(chunk: Chunk, reason: str):
    key = f"{chunk.metadata.doc_id}/{chunk.chunk_id}_rejected.json"
    body = {
        "chunk": chunk.model_dump(),
        "reason": reason,
    }

    s3.put_object(
        Bucket=REJECTED_BUCKET_NAME,
        Key=key,
        Body=json.dumps(body),
        ContentType="application/json",
    )


@retry(
    stop=stop_after_delay(300),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=before_sleep_hook,
)
def classify_chunk(chunk: Chunk) -> ChunkClassification:
    prompt = CHUNK_CLASSIFICATION_PROMPT.format(chunk_text=chunk.text)

    classification = client.chat.completions.create(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": prompt}],
        response_model=ChunkClassification,
        inferenceConfig={"maxTokens": 1024, "temperature": 0.1, "topP": 0.9},
    )

    return classification


def read_config(job_id: str) -> JobConfig:
    response = dynamodb.get_item(
        TableName=JOB_TABLE_NAME, Key={"job_id": {"S": job_id}}
    )

    if "Item" not in response:
        return JobConfig(job_id=job_id, abort=False)

    item = {k: deserializer.deserialize(v) for k, v in response["Item"].items()}
    return JobConfig(**item)


def handler(event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sqs_event = parse(event, model=SqsModel)

        if len(sqs_event.Records) != 1:
            logger.error(f"Batch size must be 1, got {len(sqs_event.Records)}")
            raise Exception("Invalid batch size")

        record = sqs_event.Records[0]
        message_data = json.loads(record.body)
        queue_message = QueueMessage(**message_data)

        config = read_config(queue_message.job_id)

        if config.abort:
            logger.info(f"Job {queue_message.job_id} is aborted, failing message")
            return {"statusCode": 500}

        try:
            classification = classify_chunk(queue_message.chunk)

            if classification.accept_chunk:
                accept_chunk(queue_message.chunk)
                logger.info(f"Accepted chunk {queue_message.chunk.chunk_id}")
            else:
                reject_chunk(queue_message.chunk, classification.reason or "Not useful")
                logger.info(
                    f"Rejected chunk {queue_message.chunk.chunk_id}: {classification.reason}"
                )

        except MaxWaitTimeReached:
            logger.error("Max wait time reached, setting abort flag")
            raise_abort_flag(queue_message.job_id)
            return {"statusCode": 500}

        except Exception as e:
            error_str = traceback.format_exc()
            logger.error(f"Error processing chunk: {error_str}")
            reject_chunk(queue_message.chunk, error_str)
            return {"statusCode": 500}

        return {"statusCode": 200}

    except Exception as e:
        logger.error(f"Handler error: {traceback.format_exc()}")
        return {"statusCode": 500}
