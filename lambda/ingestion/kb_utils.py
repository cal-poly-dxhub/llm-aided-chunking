import json
import boto3
from typing import Dict, Any, Optional
from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    doc_id: str
    source: str
    source_url: str
    chunk_index: int
    total_chunks: int


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class QueueMessage(BaseModel):
    job_id: str
    chunk: Chunk


def upload_chunk_to_s3(chunk: Chunk, target_bucket: str) -> None:
    s3_client = boto3.client("s3")

    s3_client.put_object(
        Bucket=target_bucket,
        Key=chunk.chunk_id,
        Body=chunk.text,
        ContentType="text/plain",
    )

    metadata_content = {
        "metadataAttributes": {
            "doc_id": chunk.metadata.doc_id,
            "source": chunk.metadata.source,
            "source_url": chunk.metadata.source_url,
            "chunk_index": chunk.metadata.chunk_index,
            "total_chunks": chunk.metadata.total_chunks,
        }
    }

    s3_client.put_object(
        Bucket=target_bucket,
        Key=f"{chunk.chunk_id}.metadata.json",
        Body=json.dumps(metadata_content, indent=2),
        ContentType="application/json",
    )


def queue_chunks_for_second_look(
    rejected_chunks: Dict[str, list], queue_url: str, job_id: str
) -> None:
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
