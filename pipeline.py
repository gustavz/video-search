import argparse
import json
import logging
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import librosa
import numpy as np
import openai
import torch
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from scenedetect import ContentDetector, detect
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    ClapModel,
    ClapProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

openai.api_key = os.getenv("OPENAI_API_KEY")

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────

# Model checkpoints for search embeddings - using 512-dimension models
CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"  # 512-dim CLIP for vision and text
CLAP_MODEL_NAME: str = "laion/larger_clap_music_and_speech"  # 512-dim CLAP for audio
ASR_MODEL_NAME: str = "openai/whisper-medium"  # Whisper checkpoint name

# OpenAI / reranking
OPENAI_MODEL: str = "gpt-4o"
PARAPHRASE_COUNT: int = 3  # max query rewrites
RERANK_DEPTH: int = 25  # top‑N sent to GPT for rerank

# LLM Prompt Instructions
QUERY_EXPANSION_PROMPT: str = (
    "Generate up to 5 diverse, semantically equivalent paraphrases for the "
    "multimedia search query. Preserve quoted speech verbatim. Do not include "
    "the original query in the paraphrases list."
)

RERANK_PROMPT: str = (
    "You are a video search ranking assistant. Given the user query and candidate "
    "results, reorder them from most relevant to least relevant based on the query. "
    "Return only the IDs in the correct order."
)

# ANN search
TOP_K: int = 10  # default number of hits returned
HNSW_DISTANCE: str = "Cosine"
EMBEDDING_DIM: int = 512  # Both CLIP and CLAP use 512 dimensions

# Logging config
logging.basicConfig(
    format="%(asctime)s • %(levelname)s • %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("multivideo")

# ────────────────────────────────────────────────────────────────────────────────
# DEVICE SELECTION
# ────────────────────────────────────────────────────────────────────────────────


def autodetect_device() -> torch.device:  # noqa: D401
    """Return best available torch device (cuda ▸ mps ▸ cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: torch.device = autodetect_device()
LOGGER.info("Using torch device: %s", DEVICE)

# ────────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ────────────────────────────────────────────────────────────────────────────────


class QueryExpansion(BaseModel):
    """Model for query expansion with paraphrased queries."""

    paraphrases: list[str] = Field(
        description="List of diverse, semantically equivalent paraphrases of the original query"
    )


class RankingResult(BaseModel):
    """Model for reranked search results."""

    ids: list[str] = Field(
        description="List of result IDs ordered from most relevant to least relevant"
    )


# ────────────────────────────────────────────────────────────────────────────────
# QDRANT CONFIG
# ────────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class QdrantCfg:
    host: str = "localhost"
    port: int = 6333
    collection: str = "videos"


# ────────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ────────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ModelBundle:
    # CLIP model for both text and vision encoding
    clip_tokenizer: AutoTokenizer
    clip_processor: AutoImageProcessor
    clip_model: AutoModel

    # Audio encoding (CLAP)
    clap_model: ClapModel
    clap_processor: ClapProcessor

    # Speech recognition (Whisper)
    whisper_model: WhisperForConditionalGeneration
    whisper_processor: WhisperProcessor


def load_models(device: torch.device = DEVICE) -> ModelBundle:
    """Load and return all lightweight encoders on *device*."""

    # Load CLIP model for text and vision encoding
    LOGGER.info("Loading CLIP model: %s", CLIP_MODEL_NAME)
    clip_tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = AutoImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME)
    clip_model.to(device).eval()

    # CLAP audio encoder
    LOGGER.info("Loading CLAP model: %s", CLAP_MODEL_NAME)
    clap = ClapModel.from_pretrained(CLAP_MODEL_NAME)
    clap.to(device).eval()

    # CLAP processor for audio preprocessing
    clap_proc = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)

    # Whisper for speech recognition - use CPU for MPS compatibility
    whisper_device = "cpu" if str(device) == "mps" else str(device)
    if str(device) == "mps":
        LOGGER.warning("Using CPU for Whisper due to MPS sparse tensor limitations")

    LOGGER.info("Loading Whisper model: %s", ASR_MODEL_NAME)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_NAME)
    whisper_model.to(whisper_device).eval()
    whisper_proc = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)

    return ModelBundle(
        clip_tokenizer=clip_tokenizer,
        clip_processor=clip_processor,
        clip_model=clip_model,
        clap_model=clap,
        clap_processor=clap_proc,
        whisper_model=whisper_model,
        whisper_processor=whisper_proc,
    )


# ────────────────────────────────────────────────────────────────────────────────
# VIDEO & AUDIO UTILITIES
# ────────────────────────────────────────────────────────────────────────────────


def ffmpeg_extract_audio(src: Path, dst: Path, start: float | None, end: float | None) -> None:
    """Extract mono 16 kHz PCM‑WAV using ffmpeg."""
    cmd: list[str | os.PathLike] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if start is not None:
        cmd += ["-ss", f"{start}"]
    if end is not None:
        cmd += ["-to", f"{end}"]
    cmd += [str(dst)]
    subprocess.run(cmd, check=True)


def detect_scenes(video: Path, threshold: float = 30.0) -> list[tuple[float, float]]:
    """Detect scene changes in video using PySceneDetect."""
    # Simple one-line scene detection
    scene_list = detect(str(video), ContentDetector(threshold=threshold))

    if scene_list:
        # Convert scene cuts to time intervals
        return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    else:
        # No scenes detected, use whole video duration
        cap = cv2.VideoCapture(str(video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        duration_seconds = float(total_frames) / fps
        return [(0.0, duration_seconds)]


def sample_frames(video: Path, start: float, end: float, n: int = 8) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(int(start * fps), min(int(end * fps), total - 1), n, dtype=int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def pool(vectors: Iterable[np.ndarray]) -> np.ndarray:
    return np.stack(list(vectors)).mean(axis=0)


def download_video(url: str, output_dir: Path) -> Path:
    """Download a video from URL to output_dir and return the path."""
    parsed = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed.path) or "video.mp4"
    # Ensure it has a video extension
    if not any(
        filename.lower().endswith(ext)
        for ext in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"]
    ):
        filename += ".mp4"

    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading video from %s to %s", url, output_path)
    urllib.request.urlretrieve(url, output_path)
    return output_path


def find_video_files(directory: Path) -> list[Path]:
    """Find all video files in a directory recursively."""
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}
    video_files = []

    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)

    return sorted(video_files)


def resolve_video_inputs(input_path: str) -> list[Path]:
    """Resolve input path to a list of video files.

    Args:
        input_path: Can be a file path, URL, or directory path

    Returns:
        List of video file paths to process
    """
    # Check if it's a URL
    if input_path.startswith(("http://", "https://")):
        # Download to temp directory
        temp_dir = Path(tempfile.mkdtemp())
        video_path = download_video(input_path, temp_dir)
        return [video_path]

    # Convert to Path for local files/directories
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    if path.is_file():
        return [path]

    if path.is_dir():
        video_files = find_video_files(path)
        if not video_files:
            raise ValueError(f"No video files found in directory: {input_path}")
        return video_files

    raise ValueError(f"Invalid input path: {input_path}")


# ────────────────────────────────────────────────────────────────────────────────
# QDRANT HELPERS
# ────────────────────────────────────────────────────────────────────────────────


def get_qdrant(cfg: QdrantCfg) -> QdrantClient:
    return QdrantClient(host=cfg.host, port=cfg.port)


def ensure_collection(client: QdrantClient, cfg: QdrantCfg) -> None:
    if cfg.collection in {c.name for c in client.get_collections().collections}:
        return
    vectors = {
        "text": qmodels.VectorParams(size=EMBEDDING_DIM, distance=HNSW_DISTANCE),
        "audio": qmodels.VectorParams(size=EMBEDDING_DIM, distance=HNSW_DISTANCE),
        "vision": qmodels.VectorParams(size=EMBEDDING_DIM, distance=HNSW_DISTANCE),
    }
    client.create_collection(cfg.collection, vectors_config=vectors)
    LOGGER.info("Created qdrant collection '%s'", cfg.collection)


# ────────────────────────────────────────────────────────────────────────────────
# LLM UTILITIES (always on)                                                    ─
# ────────────────────────────────────────────────────────────────────────────────


def expand_query(query: str, n: int = PARAPHRASE_COUNT) -> list[str]:
    try:
        client = openai.OpenAI()
        response = client.responses.parse(
            model=OPENAI_MODEL,
            instructions=QUERY_EXPANSION_PROMPT,
            input=query,
            text_format=QueryExpansion,
        )
        variants = response.output_parsed.paraphrases
        variants = [v for v in variants if isinstance(v, str) and v != query]
        return [query] + variants[: n - 1]
    except Exception:  # broad – might be JSON decode or API error
        LOGGER.exception("LLM query expansion failed – using original query only")
        return [query]


def llm_rerank(query: str, hits: list[dict]) -> list[dict]:
    candidates = [
        {
            "id": h["id"],
            "type": h["type"],
            "snippet": h.get("transcript", h.get("video_id", "")),
            "score": h["score"],
        }
        for h in hits[:RERANK_DEPTH]
    ]

    try:
        client = openai.OpenAI()
        response = client.responses.parse(
            model=OPENAI_MODEL,
            instructions=RERANK_PROMPT,
            input=f"Query: {query}\n\nCandidate results: {json.dumps(candidates, indent=2)}",
            text_format=RankingResult,
        )

        ordered = response.output_parsed.ids
        pos = {o: i for i, o in enumerate(ordered)}
        return sorted(hits, key=lambda h: pos.get(h["id"], 1e9))
    except Exception:
        LOGGER.exception("LLM rerank failed – falling back to ANN order")
        return hits


# ────────────────────────────────────────────────────────────────────────────────
# INGEST                                                                      ─
# ────────────────────────────────────────────────────────────────────────────────


def ingest_single_video(
    video: Path, cfg: QdrantCfg, models: ModelBundle, client: QdrantClient
) -> None:
    """Ingest a single video file."""
    vid_id = video.stem
    scenes = detect_scenes(video)
    text_vectors: list[np.ndarray] = []
    audio_vectors: list[np.ndarray] = []
    vision_vectors: list[np.ndarray] = []

    for sid, (st, et) in enumerate(tqdm(scenes, desc=f"{vid_id} scenes")):
        frames = sample_frames(video, st, et)
        vis_inp = models.clip_processor(images=frames, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            # Use CLIP vision encoding
            vvec = models.clip_model.get_image_features(**vis_inp).mean(dim=0).cpu().numpy()

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            ffmpeg_extract_audio(video, Path(tmp.name), st, et)

            # Use librosa for reliable audio loading (Whisper)
            wav_for_whisper, sr = librosa.load(tmp.name, sr=16000)

            # Process audio for Whisper
            inputs = models.whisper_processor(
                wav_for_whisper, sampling_rate=16000, return_tensors="pt"
            )

            # Generate transcription
            with torch.no_grad():
                predicted_ids = models.whisper_model.generate(inputs["input_features"])
                transcript = models.whisper_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0].strip()

            # Use librosa for CLAP audio loading (typically 48kHz)
            wav_for_clap, _ = librosa.load(tmp.name, sr=48000)
            wav = torch.from_numpy(wav_for_clap)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)  # Add batch dimension for consistency

        # text vector
        tinp = models.clip_tokenizer(
            transcript or " ", return_tensors="pt", max_length=77, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            # Use CLIP text encoding
            tvec = models.clip_model.get_text_features(**tinp).squeeze().cpu().numpy()

        # audio vector
        with torch.no_grad():
            # Use HuggingFace CLAP processor and model
            audio_inputs = models.clap_processor(
                audios=[wav.squeeze().numpy()], return_tensors="pt"
            ).to(DEVICE)
            avec = models.clap_model.get_audio_features(**audio_inputs).squeeze().cpu().numpy()

        # Store separate vectors for pooling
        text_vectors.append(tvec)
        audio_vectors.append(avec)
        vision_vectors.append(vvec)

        client.upsert(
            cfg.collection,
            [
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"text": tvec.tolist(), "audio": avec.tolist(), "vision": vvec.tolist()},
                    payload={
                        "type": "chunk",
                        "scene_key": f"{vid_id}_sc{sid:04d}",
                        "video_id": vid_id,
                        "scene_id": sid,
                        "start": round(st, 2),
                        "end": round(et, 2),
                        "duration": round(et - st, 2),
                        "transcript": transcript,
                    },
                )
            ],
        )

    # full‑video pooled vectors
    client.upsert(
        cfg.collection,
        [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "text": pool(text_vectors).tolist(),
                    "audio": pool(audio_vectors).tolist(),
                    "vision": pool(vision_vectors).tolist(),
                },
                payload={
                    "type": "full",
                    "scene_key": f"{vid_id}_full",
                    "video_id": vid_id,
                    "duration": round(scenes[-1][1], 2),
                },
            )
        ],
    )
    LOGGER.info("Ingested %s → %d chunks + full", vid_id, len(scenes))


def ingest(input_path: str, cfg: QdrantCfg) -> None:
    """Ingest video(s) from file path, URL, or directory."""
    video_files = resolve_video_inputs(input_path)

    # Load models and client once for all videos
    models = load_models()
    client = get_qdrant(cfg)
    ensure_collection(client, cfg)

    LOGGER.info("Found %d video(s) to ingest", len(video_files))

    for video in video_files:
        try:
            ingest_single_video(video, cfg, models, client)
        except Exception as e:
            LOGGER.error("Failed to ingest %s: %s", video, e)
            continue


# ────────────────────────────────────────────────────────────────────────────────
# SEARCH                                                                     ─
# ────────────────────────────────────────────────────────────────────────────────


def embed_query(query: str, models: ModelBundle) -> dict[str, np.ndarray]:
    """Create embeddings for each modality from a text query."""
    embeddings = {}

    # Text embedding using CLIP text encoder
    text_input = models.clip_tokenizer(
        query, return_tensors="pt", max_length=77, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        embeddings["text"] = (
            models.clip_model.get_text_features(**text_input).squeeze().cpu().numpy()
        )

    # Vision embedding using CLIP text encoder (cross-modal search)
    # CLIP is designed so text and vision embeddings are in the same space
    embeddings["vision"] = embeddings["text"]  # Same embedding space for cross-modal search

    # Audio embedding using CLAP text encoder
    with torch.no_grad():
        # CLAP also supports text input for cross-modal audio search
        audio_input = models.clap_processor(text=[query], return_tensors="pt").to(DEVICE)
        embeddings["audio"] = (
            models.clap_model.get_text_features(**audio_input).squeeze().cpu().numpy()
        )

    return embeddings


def search(query: str, cfg: QdrantCfg, limit: int = TOP_K) -> None:
    models = load_models()
    client = get_qdrant(cfg)

    expanded = expand_query(query)
    union: dict[str, dict] = {}

    for q in expanded:
        # Get text embedding and search text vectors
        qvec = embed_query(q, models)

        # Search across all three modalities
        for vector_name in ["text", "audio", "vision"]:
            hits = client.search(
                cfg.collection,
                (vector_name, qvec[vector_name]),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            for h in hits:
                # Deduplicate hits and aggregate scores
                entry = union.setdefault(h.id, {"payload": h.payload, "sum": 0.0, "cnt": 0})
                entry["sum"] += h.score
                entry["cnt"] += 1

    scored = []
    for uid, p in union.items():
        scored.append(
            {
                "id": uid,
                **p["payload"],
                "score": round(p["sum"] / p["cnt"], 4),
            }
        )

    scored.sort(key=lambda d: d["score"], reverse=True)

    ranked = llm_rerank(query, scored)
    print(json.dumps(ranked[:limit], indent=2))


# ────────────────────────────────────────────────────────────────────────────────
# CLI ENTRYPOINT                                                              ─
# ────────────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("multimodal video search")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ingest command
    p_ing = sub.add_parser("ingest", help="Ingest video(s) into Qdrant")
    p_ing.add_argument(
        "--input",
        required=True,
        help="Path to video file, URL, or directory containing videos",
    )
    p_ing.add_argument("--collection", default="videos", help="Qdrant collection name")

    # search command
    p_s = sub.add_parser("search", help="Search the collection with a query")
    p_s.add_argument("--query", required=True, help="Natural language query")
    p_s.add_argument("--collection", default="videos")
    p_s.add_argument("--top_k", type=int, default=TOP_K, help="Number of results to return")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = QdrantCfg(collection=args.collection)

    if args.cmd == "ingest":
        ingest(args.input, cfg)
    else:
        search(args.query, cfg, args.top_k)


if __name__ == "__main__":
    main()
