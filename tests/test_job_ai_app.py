import json
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

# Ensure job_ai_app directory and root are on sys.path
test_dir = Path(__file__).resolve().parent
app_dir = test_dir.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
if str(app_dir.parent) not in sys.path:
    sys.path.insert(0, str(app_dir.parent))

try:
    from job_ai_app.database import (
        init_db, save_job_postings, get_jobs, get_job_by_id,
        update_job_fit, save_application_kit, get_application_kit_by_job_id,
        update_job_status, compute_job_hash
    )
    from job_ai_app.llm_client import (
        extract_json_from_text, UnifiedLLMClient, GroqProvider, GeminiProvider, OllamaProvider
    )
    from job_ai_app.matcher import (
        FitReport, ApplicationKitReport, AuditReport,
        evaluate_job_fit, generate_application_kit, review_application_kit,
        heuristic_fallback_fit
    )
    from job_ai_app.scraper import fetch_job_listings
except ImportError:
    from database import (
        init_db, save_job_postings, get_jobs, get_job_by_id,
        update_job_fit, save_application_kit, get_application_kit_by_job_id,
        update_job_status, compute_job_hash
    )
    from llm_client import (
        extract_json_from_text, UnifiedLLMClient, GroqProvider, GeminiProvider, OllamaProvider
    )
    from matcher import (
        FitReport, ApplicationKitReport, AuditReport,
        evaluate_job_fit, generate_application_kit, review_application_kit,
        heuristic_fallback_fit
    )
    from scraper import fetch_job_listings


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Fixture initializing a fresh SQLite database for each test."""
    db_file = tmp_path / "test_job_ai_app.db"
    db_url = f"sqlite:///{db_file}"
    init_db(db_url)
    return db_url


def test_compute_job_hash():
    hash1 = compute_job_hash("Software Engineer", "TechCo", "Remote", "https://example.com/job/1")
    hash2 = compute_job_hash("Software Engineer", "TechCo", "Remote", "https://example.com/job/1")
    hash3 = compute_job_hash("Software Engineer", "TechCo", "Remote", None)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_database_crud_and_deduplication(setup_test_db):
    db_url = setup_test_db

    sample_jobs = [
        {
            "job_url": "https://example.com/job/101",
            "title": "Backend Python Engineer",
            "company": "Alpha Corp",
            "location": "Remote",
            "site": "linkedin",
            "job_type": "Full-time",
            "description": "Python, FastAPI, Docker",
            "min_amount": 100000,
            "max_amount": 140000,
            "currency": "USD",
            "is_remote": True,
            "match_score": 85,
            "matching_skills": ["Python", "FastAPI"],
            "missing_skills": ["Kubernetes"],
            "verdict": "STRONG_MATCH",
            "fit_summary": "Great match"
        },
        {
            "job_url": "https://example.com/job/102",
            "title": "Frontend React Engineer",
            "company": "Beta Corp",
            "location": "New York, NY",
            "site": "indeed",
            "job_type": "Full-time",
            "description": "React, TypeScript, CSS",
            "match_score": 60,
            "verdict": "APPLY"
        }
    ]

    # Save job postings
    inserted, skipped = save_job_postings(sample_jobs, db_url=db_url)
    assert inserted == 2
    assert skipped == 0

    # Attempt inserting duplicate
    inserted_dup, skipped_dup = save_job_postings([sample_jobs[0]], db_url=db_url)
    assert inserted_dup == 0
    assert skipped_dup == 1

    # Fetch jobs
    all_jobs = get_jobs(db_url=db_url)
    assert len(all_jobs) == 2
    assert all_jobs[0].match_score == 85  # Order by score desc

    # Get job by ID
    job_id = all_jobs[0].id
    fetched_job = get_job_by_id(job_id, db_url=db_url)
    assert fetched_job is not None
    assert fetched_job.title == "Backend Python Engineer"

    # Update job fit
    fit_update = {
        "match_score": 90,
        "matching_skills": ["Python", "FastAPI", "Docker"],
        "missing_skills": [],
        "verdict": "STRONG_MATCH",
        "fit_summary": "Updated fit summary"
    }
    updated_job = update_job_fit(job_id, fit_update, db_url=db_url)
    assert updated_job.match_score == 90

    # Update job status
    status_updated = update_job_status(job_id, "Applied", db_url=db_url)
    assert status_updated.status == "Applied"

    # Application Kit CRUD
    app_kit = save_application_kit(
        job_id=job_id,
        tailored_cover_letter="Dear Alpha Corp...",
        customized_bullets=["Bullet 1", "Bullet 2"],
        audit_status="Passed",
        audit_notes="No issues",
        db_url=db_url
    )
    assert app_kit.id is not None

    fetched_kit = get_application_kit_by_job_id(job_id, db_url=db_url)
    assert fetched_kit is not None
    assert fetched_kit.tailored_cover_letter == "Dear Alpha Corp..."


def test_extract_json_from_text():
    markdown_json = "```json\n{\"match_score\": 90, \"verdict\": \"STRONG_MATCH\"}\n```"
    raw_json = "{\"match_score\": 90, \"verdict\": \"STRONG_MATCH\"}"
    text_wrapped = "Here is the result: {\"match_score\": 90, \"verdict\": \"STRONG_MATCH\"} Thank you."

    assert extract_json_from_text(markdown_json) == '{"match_score": 90, "verdict": "STRONG_MATCH"}'
    assert extract_json_from_text(raw_json) == '{"match_score": 90, "verdict": "STRONG_MATCH"}'
    assert extract_json_from_text(text_wrapped) == '{"match_score": 90, "verdict": "STRONG_MATCH"}'


def test_unified_llm_client_fallback_chain():
    client = UnifiedLLMClient(primary_provider="groq")
    chain = client.get_provider_chain()
    provider_names = [name for name, _ in chain]

    assert provider_names == ["groq", "gemini", "ollama"]


def test_heuristic_fallback_fit():
    jd = "Looking for a Python software engineer experienced with FastAPI and Docker."
    resume = "Senior Software Engineer skilled in Python, FastAPI, SQL, and Docker."

    fit = heuristic_fallback_fit(jd, resume)
    assert fit.match_score >= 70
    assert "python" in fit.matching_skills
    assert fit.verdict in ["STRONG_MATCH", "APPLY"]


def test_evaluate_job_fit_fallback():
    jd = "Seeking Java Spring Boot developer with PostgreSQL experience."
    resume = "Java Developer experienced with Spring Boot and PostgreSQL."

    fit = evaluate_job_fit(jd, resume, llm_client=None)
    assert isinstance(fit, FitReport)
    assert fit.match_score >= 50


def test_drafter_and_reviewer_agents():
    jd = "Seeking Python Developer"
    resume = "Experienced Python Software Engineer"

    kit = generate_application_kit("Python Dev", "TechCorp", jd, resume)
    assert isinstance(kit, ApplicationKitReport)
    assert len(kit.tailored_cover_letter) > 50
    assert len(kit.customized_bullets) >= 1

    audit = review_application_kit(resume, jd, kit.tailored_cover_letter, kit.customized_bullets)
    assert isinstance(audit, AuditReport)
    assert audit.audit_status in ["Passed", "Needs Review", "Warning"]


@patch("job_ai_app.scraper.scrape_jobs")
def test_fetch_job_listings_mocked(mock_scrape, setup_test_db):
    db_url = setup_test_db

    mock_df = pd.DataFrame([
        {
            "job_url": "https://example.com/job/301",
            "title": "Cloud Architect",
            "company": "CloudCo",
            "location": "Remote",
            "site": "linkedin",
            "description": "AWS, Terraform, Kubernetes",
            "date_posted": "2026-09-17",
            "min_amount": 150000,
            "max_amount": 180000,
            "currency": "USD",
            "is_remote": True,
            "job_type": "fulltime"
        }
    ])
    mock_scrape.return_value = mock_df

    res = fetch_job_listings(
        search_term="Cloud Architect",
        location="Remote",
        results_wanted=1,
        db_url=db_url
    )

    assert res["total_scraped"] == 1
    assert res["inserted"] == 1
    assert res["skipped"] == 0

    saved_jobs = get_jobs(db_url=db_url)
    assert len(saved_jobs) == 1
    assert saved_jobs[0].title == "Cloud Architect"
