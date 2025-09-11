CHUNK_CLASSIFICATION_PROMPT = """Analyze this text chunk and determine if it contains useful, substantive content that should be included in a knowledge base.

Text chunk: {chunk_text}

Consider rejecting chunks that are:
- Navigation elements, headers, footers
- Copyright notices or legal boilerplate
- Empty or very short content
- Repetitive content
- Non-informative content

Accept chunks that contain:
- Substantive information
- Educational content
- Detailed explanations
- Useful data or facts"""
