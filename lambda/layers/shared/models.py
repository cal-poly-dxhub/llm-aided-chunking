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
