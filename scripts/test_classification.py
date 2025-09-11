#!/usr/bin/env python3

import boto3
import instructor
from pydantic import BaseModel
from typing import Optional


class ChunkClassification(BaseModel):
    accept_chunk: bool
    reason: Optional[str] = None


PROMPT = """Analyze this text chunk and determine if it contains useful, substantive content.

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


def test_classification():
    brt = boto3.client("bedrock-runtime")
    client = instructor.from_bedrock(brt, mode=instructor.Mode.BEDROCK_TOOLS)

    # Test chunk
    test_chunk = "<title>2025 Agricultural Assessment Guide for Wisconsin Property Owners </title>\n<headers>\n<header>XII. Agricultural Forest </header>\nState law (sec. 70.32(2)(c)1d, Wis. Stats.), defines agricultural forest as \"land that is producing or is capable of producing commercial forest products, if the land satisfies any of the following conditions:\n<list>It is contiguous to a parcel that is classified in whole as agricultural land under this subsection, if the contiguous parcel is owned by the same person that owns the land that is producing or is capable of producing commercial forest products. In this subdivision, 'contiguous' includes separated only by a road. \nIt is located on a parcel containing land classified as agricultural land in the property tax assessment on January 1, 2004, and on January 1 of the year of assessment \nIt is located on a parcel at where least 50% (by acreage) was converted to land classified as agricultural land in the property tax assessment on January 1, 2005, or thereafter\"</list>\n<headers>\n<header>Classification scenarios </header>\nThe following pages contain classification scenarios. In these scenarios, a solid line designates a parcel's boundary while a dashed line designates a change in classification within the same parcel."

    prompt = PROMPT.format(chunk_text=test_chunk)

    classification = client.chat.completions.create(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": prompt}],
        response_model=ChunkClassification,
        inferenceConfig={"maxTokens": 1024, "temperature": 0.1, "topP": 0.9},
    )  # type: ignore

    print(f"Accept: {classification.accept_chunk}")
    print(f"Reason: {classification.reason}")


if __name__ == "__main__":
    test_classification()
