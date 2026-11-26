"""Configuration for the RAGAS evaluation harness."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from backend.config import config as app_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "evaluation" / ".env", override=True)


@dataclass(frozen=True)
class EvalConfig:
    judge_model: str = os.getenv(
        "EVAL_JUDGE_MODEL", "openai/gpt-oss-120b"
    )
    judge_temperature: float = 0.0
    embedding_model: str = app_config.EMBEDDING_MODEL
    top_k: int = app_config.TOP_K
    random_seed: int = 42
    max_retries: int = 5
    retry_delay_seconds: float = 2.0
    pinecone_namespace_prefix: str = "eval-"


eval_config = EvalConfig()
