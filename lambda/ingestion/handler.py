import os
import re
import json
import boto3
import hashlib
from datetime import datetime
from botocore.config import Config
from typing import List, Dict, Tuple, Any
import logging

# NOTE: textractor is provided by a layer
from textractor.data.text_linearization_config import TextLinearizationConfig
from aws_utils import *
from table_tools import *
from reason_codes import *
from kb_utils import (
    run_second_look,
)
from models import upload_chunk_to_s3, Chunk, ChunkMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

config = Config(read_timeout=600, retries=dict(max_attempts=5))

REGION_NAME = boto3.Session().region_name
TEXTRACT_OUTPUT_BUCKET = os.getenv("TEXTRACT_OUTPUT_BUCKET")

s3 = boto3.client("s3", region_name=REGION_NAME)
bedrock_runtime = boto3.client(
    "bedrock-runtime", region_name=REGION_NAME, config=config
)


def get_debug_dir():
    """Get debug directory path and create it if needed."""
    debug_dir = "/Users/spandan/Projects/dxhub/llm-aided-chunking/debug_info"
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def save_to_s3(bucket, key, data):
    """Save data to S3 bucket."""
    s3.put_object(Bucket=bucket, Key=key, Body=data)


def id_from_content(content):
    """Generate a deterministic ID from chunk content using SHA-256 hash."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def save_data(
    data, filename, subdir, debug_mode, primary_bucket=None, secondary_bucket=None
):
    """Save data either locally (debug) or to S3 (production)."""
    if debug_mode:
        debug_dir = get_debug_dir()
        target_dir = os.path.join(debug_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)
        filepath = os.path.join(target_dir, filename)
        with open(filepath, "w") as f:
            f.write(data)
    else:
        if subdir == "final_chunks" and primary_bucket:
            save_to_s3(primary_bucket, f"chunks/{filename}", data)
        elif subdir == "removed" and secondary_bucket:
            save_to_s3(secondary_bucket, f"removed/{filename}", data)


def strip_newline(cell: Any) -> str:
    return str(cell).strip()


def sub_header_content_splitter(string: str) -> List[str]:
    pattern = re.compile(r"<<[^>]+>>")
    segments = re.split(pattern, string)
    result = []
    for segment in segments:
        if segment.strip():
            if (
                "<header>" not in segment
                and "<list>" not in segment
                and "<table>" not in segment
            ):
                segment = [x.strip() for x in segment.split("\n") if x.strip()]
                result.extend(segment)
            else:
                result.append(segment)
    return result


def split_list_items_(items: str) -> List[str]:
    parts = re.split("(<<list>><list>|</list><</list>>)", items)
    output = []
    inside_list = False
    list_item = ""

    for p in parts:
        if p == "<<list>><list>":
            inside_list = True
            list_item = p
        elif p == "</list><</list>>":
            inside_list = False
            list_item += p
            output.append(list_item)
            list_item = ""
        elif inside_list:
            list_item += p.strip()
        else:
            output.extend(p.split("\n"))
    return output


def process_document(document, local_pdf_path: str) -> Tuple[Dict, Dict]:
    config = TextLinearizationConfig(
        hide_figure_layout=True,
        hide_table_layout=True,
        title_prefix="<titles><<title>><title>",
        title_suffix="</title><</title>>",
        hide_header_layout=True,
        section_header_prefix="<headers><<header>><header>",
        section_header_suffix="</header><</header>>",
        table_prefix="<tables><table>",
        table_suffix="</table>",
        list_layout_prefix="<<list>><list>",
        list_layout_suffix="</list><</list>>",
        hide_footer_layout=True,
        hide_page_num_layout=True,
    )

    document_holder = {}
    table_page = {}
    count = 0
    unmerge_span_cells = True

    for ids, page in enumerate(document.pages):
        content = page.get_text(config=config).split("<tables>")
        document_holder[ids] = []

        for idx, item in enumerate(content):
            if "<table>" in item:
                table = document.tables[count]
                bounding_box = table.bbox
                table_pg_number = table.page
                table_base64 = get_table_base64_from_pdf(
                    local_pdf_path, table_pg_number, bounding_box
                )

                if ids in table_page:
                    table_page[ids].append(table_base64)
                else:
                    table_page[ids] = [table_base64]

                pattern = re.compile(r"<table>(.*?)(</table>)", re.DOTALL)
                data = item
                table_match = re.search(pattern, data)
                remaining_content = data[table_match.end() :] if table_match else data

                content[idx] = f"<<table>><table>{table_base64}</table><</table>>"
                count += 1

                if "<<list>>" in remaining_content:
                    output = split_list_items_(remaining_content)
                    output = [x.strip() for x in output if x.strip()]
                    document_holder[ids].extend([content[idx]] + output)
                else:
                    document_holder[ids].extend(
                        [content[idx]]
                        + [
                            x.strip()
                            for x in remaining_content.split("\n")
                            if x.strip()
                        ]
                    )
            else:
                if "<<list>>" in item and "<table>" not in item:
                    output = split_list_items_(item)
                    output = [x.strip() for x in output if x.strip()]
                    document_holder[ids].extend(output)
                else:
                    document_holder[ids].extend(
                        [x.strip() for x in item.split("\n") if x.strip()]
                    )

    page_mapping = {}
    current_page = 1
    for page in document.pages:
        page_content = page.get_text(config=config)
        page_mapping[current_page] = page_content
        current_page += 1

    flattened_list = [item for sublist in document_holder.values() for item in sublist]
    result = "\n".join(flattened_list)
    header_split = result.split("<titles>")

    return header_split, page_mapping


def chunk_document(
    header_split,
    file,
    BUCKET,
    page_mapping,
    debug_mode=False,
    primary_bucket=None,
    secondary_bucket=None,
):
    max_words = 800
    overlap_size = 50
    chunks = {}
    chunk_header_mapping = {}

    roman_section_pattern = re.compile(r"^(?:[IVXLCDM]+)\.", re.MULTILINE)
    subsection_pattern = re.compile(r"^[A-Z]\.", re.MULTILINE)
    numbered_pattern = re.compile(r"^\d+\.", re.MULTILINE)

    doc_id = os.path.basename(file)
    logging_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def flush_chunk(chunks_list, chunk):
        if chunk["content"]:
            chunks_list.append(chunk.copy())
        return {"content": [], "metadata": {}}

    def find_page_number(content):
        for page_num, page_content in page_mapping.items():
            if content in page_content:
                return page_num
        return None

    for title_ids, items in enumerate(header_split):
        title_chunks = []
        current_chunk = {"content": [], "metadata": {}}
        num_words = 0
        chunk_counter = 0
        last_known_page = 1

        lines = sub_header_content_splitter(items)

        for line in lines:
            if not line.strip():
                continue

            page_number = find_page_number(line)
            if page_number:
                last_known_page = page_number

            current_chunk["metadata"]["page_start"] = last_known_page
            current_chunk["metadata"]["page_end"] = last_known_page
            current_chunk["metadata"]["doc"] = doc_id

            if roman_section_pattern.match(line):
                current_chunk = flush_chunk(title_chunks, current_chunk)
                current_chunk["metadata"]["section"] = line.strip()
                continue

            if subsection_pattern.match(line):
                current_chunk = flush_chunk(title_chunks, current_chunk)
                current_chunk["metadata"]["subsection"] = line.strip()
                continue

            if numbered_pattern.match(line):
                current_chunk = flush_chunk(title_chunks, current_chunk)
                current_chunk["metadata"]["sub_subsection"] = line.strip()
                continue

            if "flowchart" in line.lower():
                current_chunk["content"].append(line)
                current_chunk["metadata"]["special"] = "flowchart"
                current_chunk = flush_chunk(title_chunks, current_chunk)
                continue

            if "<table>" in line:
                current_chunk["content"].append(line)
                current_chunk["metadata"]["special"] = "table"
                current_chunk = flush_chunk(title_chunks, current_chunk)
                continue

            next_num_words = num_words + len(re.findall(r"\w+", line))
            if next_num_words > max_words:
                flat = " ".join(current_chunk["content"])
                words = re.findall(r"\w+", flat)
                overlap = words[-overlap_size:] if len(words) >= overlap_size else words
                overlap_text = " ".join(overlap)

                current_chunk["content"].append(overlap_text)
                title_chunks.append(current_chunk)
                current_chunk = {
                    "content": [overlap_text],
                    "metadata": {
                        "doc": doc_id,
                        "page_start": last_known_page,
                        "page_end": last_known_page,
                    },
                }
                num_words = len(overlap)
                chunk_counter += 1

            current_chunk["content"].append(line)
            num_words = next_num_words

        if current_chunk["content"]:
            title_chunks.append(current_chunk)

        chunks[title_ids] = title_chunks
        chunk_header_mapping[title_ids] = lines

    # Save page maps if in debug mode
    if debug_mode:
        page_maps_data = json.dumps(chunk_header_mapping, indent=4)
        save_data(
            page_maps_data,
            f"{doc_id}_{logging_timestamp}.json",
            "page_maps",
            debug_mode,
        )

    return {
        "chunks": chunks,
        "chunk_header_mapping": chunk_header_mapping,
        "doc_id": doc_id,
    }


def parse_s3_uri(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")
    s3_path = s3_uri[5:]
    bucket_name, *key_parts = s3_path.split("/", 1)
    file_key = key_parts[0] if key_parts else ""
    return bucket_name, file_key


def extract_clean_plaintext(
    doc_chunks, doc_id=None
) -> Tuple[List[str], Dict[str, List[dict]]]:
    all_cleaned_content = []
    removed_chunks = {}

    HARD_LOWER_THRESHOLD = 20
    SOFT_LOWER_THRESHOLD = 100

    junk_phrases = {
        "top of this section",
        "section header",
        "footer",
        "page x",
        "click here",
        "back to top",
    }

    def clean_line(line):
        if not isinstance(line, str):
            return ""
        return re.sub(r"<[^>]+>", "", line).strip()

    def is_junk_chunk(lines):
        cleaned = [clean_line(line).lower() for line in lines]
        return all(line in junk_phrases or not line for line in cleaned)

    def is_sentence(line):
        return line.endswith((".", "?", "!")) and len(line.split()) >= 5

    def is_gibberish(line):
        words = line.split()
        real_words = [w for w in words if re.search(r"[aeiouAEIOU]", w) and len(w) > 2]
        return len(real_words) < max(3, len(words) * 0.4)

    def log_removed_chunk(chunk, chunk_num, reason, section_id=None):
        if reason not in removed_chunks:
            removed_chunks[reason] = []
        removed_chunks[reason].append(
            {
                "text": "\n".join(chunk["content"])
                if isinstance(chunk.get("content"), list)
                else str(chunk.get("content", "")),
                "metadata": chunk.get("metadata", {}),
                "doc_id": doc_id,
                "chunk_num": chunk_num,
                "section_id": section_id,
                "removal_reason": reason,
            }
        )

    for chunk_num, chunk_group in doc_chunks.items():
        valid_lines = []

        for chunk_idx, chunk in enumerate(chunk_group):
            if not isinstance(chunk, dict) or "content" not in chunk:
                log_removed_chunk(chunk, f"{chunk_num}_{chunk_idx}", INVALID_STRUCTURE)
                continue

            chunk_content = chunk["content"]
            if not isinstance(chunk_content, list):
                log_removed_chunk(chunk, f"{chunk_num}_{chunk_idx}", INVALID_CONTENT)
                continue

            if is_junk_chunk(chunk_content):
                log_removed_chunk(chunk, f"{chunk_num}_{chunk_idx}", JUNK_CHUNK)
                continue

            original_lines = [
                clean_line(line) for line in chunk_content if clean_line(line)
            ]
            cleaned_lines = [line for line in original_lines if not is_gibberish(line)]

            if not cleaned_lines:
                log_removed_chunk(
                    chunk, f"{chunk_num}_{chunk_idx}", EMPTY_AFTER_CLEANING
                )
                continue

            total_words = sum(len(line.split()) for line in cleaned_lines)
            sentence_count = sum(1 for line in cleaned_lines if is_sentence(line))
            avg_sentence_length = total_words / max(sentence_count, 1)

            if total_words < HARD_LOWER_THRESHOLD:
                log_removed_chunk(
                    chunk,
                    f"{chunk_num}_{chunk_idx}",
                    "HARD_LOWER_THRESHOLD",
                )
                continue

            if total_words < SOFT_LOWER_THRESHOLD:
                log_removed_chunk(
                    chunk, f"{chunk_num}_{chunk_idx}", SOFT_THRESHOLD_FAIL
                )
                continue

            if avg_sentence_length < 6:
                log_removed_chunk(
                    chunk, f"{chunk_num}_{chunk_idx}", SHORT_SENTENCE_LENGTH
                )
                continue
                log_removed_chunk(
                    chunk,
                    f"{chunk_num}_{chunk_idx}",
                    f"Short average sentence length - {avg_sentence_length:.1f}",
                )
                continue

            valid_lines.extend(cleaned_lines)

        if valid_lines:
            content = "\n\n".join(valid_lines).strip()
            all_cleaned_content.append(content)

    return all_cleaned_content, removed_chunks


def process_pdf_from_s3(
    bucket_name: str,
    s3_file_path: str,
    document_url: str = "n/a",
    debug_mode=False,
    primary_bucket=None,
    secondary_bucket=None,
) -> tuple:
    s3_uri = f"s3://{bucket_name}/{s3_file_path}"

    if not TEXTRACT_OUTPUT_BUCKET:
        raise ValueError("TEXTRACT_OUTPUT_BUCKET environment variable is not set.")

    logging_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    textract_output_path = None
    try:
        document, local_pdf_path, textract_output_path = extract_textract_data(
            s3, s3_uri, bucket_name, TEXTRACT_OUTPUT_BUCKET
        )

        header_split, page_mapping = process_document(document, local_pdf_path)
        doc_chunks = chunk_document(
            header_split,
            s3_file_path,
            bucket_name,
            page_mapping,
            debug_mode,
            primary_bucket,
            secondary_bucket,
        )

        # Save raw chunks (debug only)
        if debug_mode:
            raw_chunks_data = []
            for section_id, chunk_group in doc_chunks["chunks"].items():
                for chunk in chunk_group:
                    record = {
                        "text": "\n".join(chunk["content"]),
                        "metadata": chunk.get("metadata", {}),
                        "doc_id": doc_chunks["doc_id"],
                        "section_id": section_id,
                    }
                    raw_chunks_data.append(json.dumps(record, indent=2))

            raw_chunks_content = "\n".join(raw_chunks_data)
            save_data(
                raw_chunks_content,
                f"{os.path.basename(s3_file_path)}_{logging_timestamp}.jsonl",
                "raw_chunks",
                debug_mode,
            )

        cleaned_text_chunks, removed_chunks = extract_clean_plaintext(
            doc_chunks["chunks"], doc_id=doc_chunks["doc_id"]
        )

        # Save cleaned chunks (debug only)
        if debug_mode:
            cleaned_chunks_data = []
            for chunk in cleaned_text_chunks:
                record = {
                    "text": chunk,
                    "metadata": {
                        "doc_id": doc_chunks["doc_id"],
                        "source": s3_file_path,
                        "source_url": document_url,
                    },
                }
                cleaned_chunks_data.append(json.dumps(record, indent=2))

            cleaned_chunks_content = "\n".join(cleaned_chunks_data)
            save_data(
                cleaned_chunks_content,
                f"{os.path.basename(s3_file_path)}_{logging_timestamp}.jsonl",
                "cleaned_chunks",
                debug_mode,
            )

        # Save final chunks individually to S3
        if not debug_mode and primary_bucket:
            for idx, chunk in enumerate(cleaned_text_chunks):
                chunk_obj = Chunk(
                    chunk_id=id_from_content(chunk),
                    text=chunk,
                    metadata=ChunkMetadata(
                        doc_id=doc_chunks["doc_id"],
                        source=s3_file_path,
                        source_url=document_url,
                        chunk_index=idx,
                        total_chunks=len(cleaned_text_chunks),
                    ),
                )
                upload_chunk_to_s3(chunk_obj, primary_bucket)

        # Save final chunks for debug mode
        if debug_mode:
            final_chunks_data = []
            for idx, chunk in enumerate(cleaned_text_chunks):
                record = {
                    "chunk_id": id_from_content(chunk),
                    "text": chunk,
                    "metadata": {
                        "doc_id": doc_chunks["doc_id"],
                        "source": s3_file_path,
                        "source_url": document_url,
                        "chunk_index": idx,
                        "total_chunks": len(cleaned_text_chunks),
                    },
                }
                final_chunks_data.append(json.dumps(record, indent=2))

            final_chunks_content = "\n".join(final_chunks_data)
            save_data(
                final_chunks_content,
                f"{os.path.basename(s3_file_path)}_{logging_timestamp}.jsonl",
                "final_chunks",
                debug_mode,
                primary_bucket,
                secondary_bucket,
            )

            # Save removed chunks for debug mode only
            removed_chunks_data = []
            for reason, chunks in removed_chunks.items():
                for chunk in chunks:
                    chunk_with_reason = {**chunk, "removal_reason": reason}
                    removed_chunks_data.append(json.dumps(chunk_with_reason, indent=2))

            removed_chunks_content = "\n".join(removed_chunks_data)
            save_data(
                removed_chunks_content,
                f"{os.path.basename(s3_file_path)}_{logging_timestamp}.jsonl",
                "removed",
                debug_mode,
                primary_bucket,
                secondary_bucket,
            )

        chunk_map = []
        for idx, chunk in enumerate(cleaned_text_chunks):
            chunk_map.append(
                {
                    "chunk_id": id_from_content(chunk),
                    "text": chunk,
                    "metadata": {
                        "doc_id": doc_chunks["doc_id"],
                        "source": s3_file_path,
                        "source_url": document_url,
                        "chunk_index": idx,
                        "total_chunks": len(cleaned_text_chunks),
                    },
                }
            )

        return chunk_map, removed_chunks
    finally:
        if textract_output_path:
            media_bucket, prefix = parse_s3_uri(textract_output_path)
            delete_s3_prefix(s3, media_bucket, prefix)


def lambda_handler(event, context):
    """
    Lambda handler for PDF processing.

    Expected event format:
    {
        "bucket_name": "wisconsin-chatbot-sources",
        "s3_key": "sources/wi-statute-ch75/wi-statute-ch75.pdf",
        "document_url": "optional_source_url",
        "debug_mode": false,
        "primary_output_bucket": "chunks-output-bucket",
        "secondary_output_bucket": "logs-output-bucket"
    }
    """
    try:
        bucket_name = event["bucket_name"]
        s3_key = event["s3_key"]
        document_url = event.get("document_url", "n/a")
        debug_mode = event.get("debug_mode", False)
        primary_bucket = event.get("primary_output_bucket")
        secondary_bucket = event.get("secondary_output_bucket")
        target_bucket = event.get("target_bucket")
        # job_id = event.get("job_id")
        # queue_url = os.getenv("CHUNK_QUEUE_URL")

        logger.info(f"Processing document: s3://{bucket_name}/{s3_key}")
        logger.info(f"Target bucket: {target_bucket}, Debug mode: {debug_mode}")
        logger.info(f"Intermediate textract bucket: {TEXTRACT_OUTPUT_BUCKET}")

        # Process chunks with first-pass heuristics
        chunk_map, rejected_chunks = process_pdf_from_s3(
            bucket_name,
            s3_key,
            document_url,
            debug_mode,
            primary_bucket,
            secondary_bucket,
        )

        # Upload accepted chunks to target bucket
        if target_bucket and not debug_mode:
            for chunk_data in chunk_map:
                chunk = Chunk(
                    chunk_id=chunk_data["chunk_id"],
                    text=chunk_data["text"],
                    metadata=ChunkMetadata(**chunk_data["metadata"]),
                )
                upload_chunk_to_s3(chunk, target_bucket)

        # Process rejected chunks with LLM classification
        second_look_map, final_rejected = run_second_look(
            rejected_chunks,
        )

        # Log LLM removed chunks and processing errors
        llm_removed_chunks = {}
        processing_error_chunks = {}
        
        for reason, chunks in final_rejected.items():
            if reason == "PROCESSING_ERROR":
                processing_error_chunks[reason] = chunks
            elif reason not in rejected_chunks:  # New rejections from LLM
                llm_removed_chunks[reason] = chunks

        # Save LLM removed chunks
        if llm_removed_chunks:
            logging_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            llm_removed_data = []
            for reason, chunks in llm_removed_chunks.items():
                for chunk in chunks:
                    chunk_with_reason = {**chunk, "removal_reason": reason}
                    llm_removed_data.append(json.dumps(chunk_with_reason, indent=2))
            
            llm_removed_content = "\n".join(llm_removed_data)
            save_data(
                llm_removed_content,
                f"{os.path.basename(s3_key)}_{logging_timestamp}.jsonl",
                "removed_llm",
                debug_mode,
                primary_bucket,
                secondary_bucket,
            )

        # Save processing error chunks
        if processing_error_chunks:
            logging_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            error_data = []
            for reason, chunks in processing_error_chunks.items():
                for chunk in chunks:
                    chunk_with_reason = {**chunk, "removal_reason": reason}
                    error_data.append(json.dumps(chunk_with_reason, indent=2))
            
            error_content = "\n".join(error_data)
            save_data(
                error_content,
                f"{os.path.basename(s3_key)}_{logging_timestamp}.jsonl",
                "processing_error",
                debug_mode,
                primary_bucket,
                secondary_bucket,
            )

        # Upload accepted chunks from second look
        if target_bucket and not debug_mode:
            for chunk_data in second_look_map:
                chunk = Chunk(
                    chunk_id=chunk_data["chunk_id"],
                    text=chunk_data["text"],
                    metadata=ChunkMetadata(**chunk_data["metadata"]),
                )
                upload_chunk_to_s3(chunk, target_bucket)

        for chunk_data in second_look_map:
            chunk_map.append(chunk_data)
        rejected_chunks = final_rejected

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "chunks": chunk_map,
                    "total_chunks": len(chunk_map),
                    "document": s3_key,
                    "debug_mode": debug_mode,
                }
            ),
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": str(e), "document": event.get("s3_key", "unknown")}
            ),
        }
