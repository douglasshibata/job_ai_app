"""Vector search and semantic pre-filtering service using FastEmbed & Cosine Similarity."""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

from core.database import (
    get_db_session,
    init_db,
    update_job_semantic_score,
    JobListing
)
from core.config import settings

logger = logging.getLogger(__name__)

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

class VectorSearchService:
    _embedding_model: Optional[Any] = None

    @classmethod
    def get_embedding_model(cls):
        if cls._embedding_model is None and TextEmbedding is not None:
            try:
                cls._embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
            except Exception as e:
                logger.warning(f"Failed to load FastEmbed model: {e}")
                cls._embedding_model = None
        return cls._embedding_model

    @classmethod
    def generate_embedding(cls, text: str) -> List[float]:
        model = cls.get_embedding_model()
        if model is None:
            return [0.0] * settings.VECTOR_DIMENSION

        try:
            cleaned = text[:2000].strip()
            embeddings = list(model.embed([cleaned]))
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error computing text embedding: {e}")
            return [0.0] * settings.VECTOR_DIMENSION

    @classmethod
    def rank_jobs_by_resume(
        cls,
        resume_text: str,
        jobs: List[JobListing],
        persist_scores: bool = True,
        db_url: Optional[str] = None
    ) -> List[Tuple[JobListing, float]]:
        if not jobs or not resume_text:
            return [(job, job.semantic_score or 0.0) for job in jobs]

        resume_vec = np.array(cls.generate_embedding(resume_text), dtype=np.float32)

        ranked = []
        for job in jobs:
            if job.embedding_json:
                try:
                    job_vec = np.array(json.loads(job.embedding_json), dtype=np.float32)
                except Exception:
                    job_vec = np.array(cls.generate_embedding(job.description or job.title), dtype=np.float32)
            else:
                text_to_embed = f"{job.title} at {job.company}. {job.description or ''}"
                vec_list = cls.generate_embedding(text_to_embed)
                job_vec = np.array(vec_list, dtype=np.float32)

            sim = compute_cosine_similarity(resume_vec, job_vec)
            normalized_score = round(max(0.0, min(100.0, sim * 100.0)), 1)
            job.semantic_score = normalized_score

            if persist_scores and job.id:
                update_job_semantic_score(
                    job_id=job.id,
                    semantic_score=normalized_score,
                    embedding=job_vec.tolist(),
                    db_url=db_url
                )

            ranked.append((job, normalized_score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    @classmethod
    def get_top_percentile_jobs(
        cls,
        ranked_jobs: List[Tuple[JobListing, float]],
        percentile: float = 0.15
    ) -> List[JobListing]:
        if not ranked_jobs:
            return []
        count = max(1, int(len(ranked_jobs) * percentile))
        return [job for job, _ in ranked_jobs[:count]]
