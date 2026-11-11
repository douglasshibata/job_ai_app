"""Unit tests for vector search and cosine similarity pre-filtering."""
import pytest
import numpy as np
from services.vector_search import VectorSearchService, compute_cosine_similarity
from models.entities import JobListing
from core.database import init_db

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_vector.db"
    db_url = f"sqlite:///{db_file}"
    init_db(db_url)
    return db_url

def test_cosine_similarity_computation():
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([1.0, 0.0, 0.0])
    vec_c = np.array([0.0, 1.0, 0.0])

    assert pytest.approx(compute_cosine_similarity(vec_a, vec_b), 0.01) == 1.0
    assert pytest.approx(compute_cosine_similarity(vec_a, vec_c), 0.01) == 0.0

def test_vector_search_service_embedding_generation():
    text = "Machine Learning Engineer specialized in PyTorch, transformers, and LLM fine-tuning."
    emb = VectorSearchService.generate_embedding(text)

    assert isinstance(emb, list)
    assert len(emb) == 384

def test_rank_jobs_by_resume(test_db):
    resume = "Senior Python Backend Developer with FastAPI, Docker, and PostgreSQL experience."

    job_relevant = JobListing(
        id=1,
        fingerprint="fp1",
        title="Python Backend Developer",
        company="AlphaCorp",
        description="We need a Python developer experienced with FastAPI and Docker.",
        job_url="https://example.com/1",
        source_board="linkedin",
        status="NEW"
    )

    job_irrelevant = JobListing(
        id=2,
        fingerprint="fp2",
        title="Nurse Practitioner",
        company="HealthCare",
        description="Patient care, triage, and medical examinations in clinic.",
        job_url="https://example.com/2",
        source_board="indeed",
        status="NEW"
    )

    ranked = VectorSearchService.rank_jobs_by_resume(
        resume_text=resume,
        jobs=[job_irrelevant, job_relevant],
        persist_scores=False,
        db_url=test_db
    )

    assert len(ranked) == 2
    top_job, top_score = ranked[0]
    assert top_job.title == "Python Backend Developer"
    assert top_score > ranked[1][1]

    top_15 = VectorSearchService.get_top_percentile_jobs(ranked, percentile=0.5)
    assert len(top_15) == 1
    assert top_15[0].title == "Python Backend Developer"
