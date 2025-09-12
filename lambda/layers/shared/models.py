import json
import boto3
from pydantic import BaseModel
from typing import Optional


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
    """Record in the SQS chunk queue"""

    job_id: str
    chunk: Chunk


class JobConfig(BaseModel):
    job_id: str
    abort: bool


class ClassificationResult(BaseModel):
    accept: bool
    reason: Optional[str] = None


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
