# Video Search System

## Task

Search full videos and video scenes with natural language

## Tools

1. Decoding/ Audio Extraction --> FFmpeg: https://github.com/FFmpeg/FFmpeg
2. Transcription --> Whisper: https://github.com/openai/whisper
3. Scene Detection --> PySceneDetect: https://github.com/Breakthrough/PySceneDetect
4. Video Understanding --> CLIP: https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b
5. Audio Understanding --> CLAP: https://github.com/LAION-AI/CLAP
6. Vector Store --> Qdrant: https://github.com/qdrant/qdrant
7. Query Expansion & Reranking --> OpenAI GPT-4o with structured outputs using Pydantic models

## Ideas

- Single VDB for full video and chunks embeddings but with different metadata, e.g. “full_video”, “video_chunk”
- Perform Search on multiple modalities in parallel and do late fusion / ranking

## Setup

```bash
# Install dependencies
uv sync

# Ingest a single video file
uv run python pipeline.py ingest --input /path/to/video.mp4 --collection my_videos

# Ingest all videos from a directory
uv run python pipeline.py ingest --input /path/to/video/directory/ --collection my_videos

# Ingest a video from URL
uv run python pipeline.py ingest --input https://example.com/video.mp4 --collection my_videos

# Search videos
uv run python pipeline.py search --query "person running in the rain" --collection my_videos

# Install with dev dependencies
uv sync --group dev

## Development

```bash
# Format code
uv run ruff format .

# Auto-fix linting issues
uv run ruff check --fix .

# Type checking
uv run mypy pipeline.py


```