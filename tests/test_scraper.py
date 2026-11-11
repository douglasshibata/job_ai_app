"""Unit tests for ResilientScraperService: error handling, deduplication, and sanitization."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from services.scraper_engine import ResilientScraperService
from core.database import init_db, get_job_listings, JobListing

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_scraper.db"
    db_url = f"sqlite:///{db_file}"
    init_db(db_url)
    return db_url

@patch("services.scraper_engine.scrape_jobs")
def test_scraper_board_failure_resilience(mock_scrape, test_db):
    def side_effect(site_name, **kwargs):
        if site_name == ["linkedin"]:
            raise RuntimeError("HTTP 403 Forbidden Cloudflare challenge")
        return pd.DataFrame([{
            "title": "Backend Python Developer",
            "company": "IndeedCo",
            "location": "Remote",
            "description": "<p>Build high-scale systems with <b>FastAPI</b>.</p>",
            "job_url": "https://indeed.com/job/1",
            "site": "indeed",
            "min_amount": 120000,
            "max_amount": 150000,
            "currency": "USD",
            "is_remote": True
        }])

    mock_scrape.side_effect = side_effect

    results = ResilientScraperService.execute_search(
        search_term="Python",
        location="Remote",
        results_wanted=5,
        sites=["linkedin", "indeed"],
        db_url=test_db
    )

    assert len(results) == 1
    assert results[0]["company"] == "IndeedCo"
    assert "<p>" not in results[0]["description"]
    assert "FastAPI" in results[0]["description"]

@patch("services.scraper_engine.scrape_jobs")
def test_scraper_atomic_deduplication(mock_scrape, test_db):
    mock_df = pd.DataFrame([
        {
            "title": "Staff Engineer",
            "company": "TechCorp",
            "location": "Remote",
            "description": "Distributed systems engineering",
            "job_url": "https://example.com/staff-1",
            "site": "glassdoor"
        },
        {
            "title": "Staff Engineer",
            "company": "TechCorp",
            "location": "Remote",
            "description": "Distributed systems engineering (duplicate cross-board)",
            "job_url": "https://example.com/staff-2",
            "site": "zip_recruiter"
        }
    ])
    mock_scrape.return_value = mock_df

    results_first = ResilientScraperService.execute_search(
        search_term="Staff Engineer",
        location="Remote",
        sites=["glassdoor"],
        db_url=test_db
    )
    assert len(results_first) == 1

    results_second = ResilientScraperService.execute_search(
        search_term="Staff Engineer",
        location="Remote",
        sites=["glassdoor"],
        db_url=test_db
    )
    assert len(results_second) == 0

    all_jobs = get_job_listings(db_url=test_db)
    assert len(all_jobs) == 1
