import boto3
from models import Chunk, QueueMessage


def queue_chunks_for_second_look(
    rejected_chunks: dict, queue_url: str, job_id: str
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
