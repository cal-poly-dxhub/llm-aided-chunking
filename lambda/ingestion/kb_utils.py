import boto3
from typing import Optional
from pydantic import BaseModel
import json
from models import Chunk, QueueMessage
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_exception,
)
import logging
import instructor
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
)

logger = logging.getLogger()

CHUNK_CLASSIFICATION_PROMPT = """Analyze this text chunk and determine if it contains useful, substantive content.

Text chunk: {chunk_text}

Consider rejecting chunks that are:
- Navigation elements, headers, footers
- Copyright notices or legal boilerplate
- Mostly punctuation, special characters, decorative formatting
- Empty or very short content
- Gibberish or nonsensical text
- Titles without supporting content

Accept chunks that contain:
- Substantive information
- Educational content
- Detailed explanations
- Solid paragraphs of information
- Meaningful lists or data

Keep your reasoning concise: under 50 words.
"""


class MaxWaitTimeReached(Exception):
    pass


class ChunkClassification(BaseModel):
    accept_chunk: bool
    reason: Optional[str] = None


def is_retryable_error(exc: Exception) -> bool:
    # Never retry parameter or schema validation errors
    from botocore.exceptions import ParamValidationError

    if isinstance(exc, ParamValidationError):
        return False

    # Network / connection issues are retryable
    if isinstance(exc, EndpointConnectionError):
        return True

    # If it's a botocore ClientError, inspect the response
    if isinstance(exc, ClientError):
        error_code = None
        try:
            error_code = exc.response.get("Error", {}).get("Code")
        except Exception:
            error_code = None

        # AWS sometimes also provides HTTP status codes
        status_code = None
        try:
            status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        except Exception:
            status_code = None

        # Retry for Bedrock ThrottlingException
        if error_code == "ThrottlingException":
            return True

        # Also retry for 429 too many requests (rate limits)
        if status_code == 429:
            return True

        # Maybe also retry service unavailable
        if status_code == 503:
            return True

    return False


def before_sleep_hook(retry_state):
    wait_time = retry_state.next_action.sleep
    if wait_time > 30:
        raise MaxWaitTimeReached("Maximum wait time exceeded")


@retry(
    stop=stop_after_delay(60),
    wait=wait_exponential_jitter(initial=1, exp_base=2, max=30, jitter=2),
    before_sleep=before_sleep_hook,
    retry=retry_if_exception(is_retryable_error),  # type: ignore
)
def classify_chunk(chunk_text: str) -> ChunkClassification:
    logger.info("Classifying chunk with LLM")

    bedrock = boto3.client("bedrock-runtime")
    client = instructor.from_bedrock(bedrock)

    prompt = CHUNK_CLASSIFICATION_PROMPT.format(chunk_text=chunk_text)

    response = client.chat.completions.create(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": prompt}],
        response_model=ChunkClassification,
    )

    logger.info(
        f"LLM classification response: {json.dumps(response.model_dump(), indent=2)} for chunk text starting with: {chunk_text[:200]}..."
    )

    return response


def run_second_look(rejected_chunks: dict) -> tuple[list, dict]:
    """
    Process rejected chunks through LLM classification.
    Returns (chunk_map, final_rejected_chunks)
    """
    import traceback

    chunk_map = []
    final_rejected_chunks = {}

    # Only process SOFT_THRESHOLD_FAIL chunks
    if "SOFT_THRESHOLD_FAIL" not in rejected_chunks:
        return chunk_map, rejected_chunks

    for chunk_data in rejected_chunks["SOFT_THRESHOLD_FAIL"]:
        try:
            chunk_text = chunk_data.get("text", "")
            if not chunk_text:
                continue
            
            # Clean lines before LLM classification
            import re
            def clean_line(line):
                if not isinstance(line, str):
                    return ""
                return re.sub(r"<[^>]+>", "", line).strip()
            
            lines = chunk_text.split('\n')
            cleaned_lines = [clean_line(line) for line in lines if clean_line(line)]
            cleaned_text = '\n'.join(cleaned_lines)
            
            if not cleaned_text:
                continue

            # Classify cleaned chunk using LLM
            classification = classify_chunk(cleaned_text)

            if classification.accept_chunk:
                # Add to chunk map for upload by main handler
                chunk_map.append(chunk_data)
                logger.info(f"Accepted chunk {chunk_data.get('chunk_id', 'unknown')}")

            else:
                # Log as rejected
                reason = classification.reason or "LLM_REJECTED"
                if reason not in final_rejected_chunks:
                    final_rejected_chunks[reason] = []

                final_rejected_chunks[reason].append(
                    {**chunk_data, "llm_rejection_reason": classification.reason}
                )

                logger.info(
                    f"Rejected chunk {chunk_data.get('chunk_id', 'unknown')}: {reason}"
                )

        except MaxWaitTimeReached:
            logger.error("Max wait time reached, aborting second look processing")
            break
        except Exception as e:
            error_str = traceback.format_exc()
            logger.error(f"Error processing chunk: {error_str}")
            # Log as error
            if "PROCESSING_ERROR" not in final_rejected_chunks:
                final_rejected_chunks["PROCESSING_ERROR"] = []
            final_rejected_chunks["PROCESSING_ERROR"].append(
                {**chunk_data, "error": str(e), "stack_trace": error_str}
            )

    # Add other rejection reasons to final rejected chunks
    for reason, chunks in rejected_chunks.items():
        if reason != "SOFT_THRESHOLD_FAIL":
            final_rejected_chunks[reason] = chunks

    return chunk_map, final_rejected_chunks


def queue_for_second_look(rejected_chunks: dict, queue_url: str, job_id: str) -> None:
    QUEUE_REJECTION_REASONS = ["SOFT_THRESHOLD_FAIL"]

    sqs_client = boto3.client("sqs")

    for reason in QUEUE_REJECTION_REASONS:
        if reason in rejected_chunks:
            for chunk_data in rejected_chunks[reason]:
                chunk = Chunk(**chunk_data)
                message = QueueMessage(job_id=job_id, chunk=chunk)

                sqs_client.send_message(
                    QueueUrl=queue_url, MessageBody=message.model_dump_json()
                )
