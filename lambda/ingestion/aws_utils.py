from typing import List
import os
from textractor import Textractor
from textractor.data.constants import TextractFeatures
import boto3

# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

REGION_NAME = boto3.Session().region_name
print("region", REGION_NAME)
# OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')


def get_emb(embeddings_client, passage: str) -> List[float]:
    """Get embedding for a given passage using titan embeddings."""

    # Invoke the model
    embedding = embeddings_client.embed_query(passage)
    return embedding


def extract_textract_data_local(local_pdf_path: str):
    """
    Run Textract on a local PDF file (synchronously).
    Returns:
      - document: Textractor Document object
      - local_pdf_path: echo of input path (for downstream helpers)
      - None: placeholder for textract_output_path (unused locally)
    """
    extractor = Textractor(region_name=REGION_NAME)

    # Use analyze_document for local files (synchronous)
    document = extractor.analyze_document(
        file_source=local_pdf_path,
        features=[TextractFeatures.LAYOUT, TextractFeatures.TABLES],
        save_image=False,
    )

    return document, local_pdf_path, None


def extract_textract_data(s3, s3_file, bucket_name, media_bucket_name):
    """Extract structured text data using Textract."""

    extractor = Textractor(region_name=REGION_NAME)

    file_name, ext = os.path.splitext(os.path.basename(s3_file))
    textract_output_path = f"s3://{media_bucket_name}/textract-output/{file_name}/"

    document = extractor.start_document_analysis(
        file_source=s3_file,
        features=[TextractFeatures.LAYOUT, TextractFeatures.TABLES],
        save_image=True,
        s3_output_path=textract_output_path,
    )

    print("Document analysis started... ")

    # Download pdf from s3
    os.makedirs("/tmp/pdf", exist_ok=True)
    local_pdf_path = f"/tmp/pdf/{os.path.basename(file_name)}.pdf"
    download_from_s3(s3, s3_file, local_pdf_path)

    return document, local_pdf_path, textract_output_path


def download_from_s3(s3, s3_path, local_path):
    s3_bucket, s3_key = s3_path.replace("s3://", "").split("/", 1)
    s3.download_file(s3_bucket, s3_key, local_path)


def delete_s3_prefix(s3, bucket_name, prefix):
    """Deletes all objects under a given prefix in an S3 bucket."""
    try:
        objects_to_delete = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        delete_keys = {"Objects": []}
        if "Contents" in objects_to_delete:
            delete_keys["Objects"] = [
                {"Key": obj["Key"]} for obj in objects_to_delete["Contents"]
            ]
            if delete_keys["Objects"]:
                s3.delete_objects(Bucket=bucket_name, Delete=delete_keys)
                print(
                    f"Successfully deleted temporary Textract files from s3://{bucket_name}/{prefix}"
                )
    except Exception as e:
        print(f"Error deleting files from s3://{bucket_name}/{prefix}: {e}")
