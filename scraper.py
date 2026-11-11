"""Backward compatibility layer for scraper function."""
from typing import List, Dict, Any, Optional
import pandas as pd
try:
    from jobspy import scrape_jobs
except ImportError:
    scrape_jobs = None

from services.scraper_engine import ResilientScraperService
from core.database import compute_job_hash
from core.config import settings

def fetch_job_listings(
    search_term: str = settings.DEFAULT_SEARCH_TERM,
    location: str = settings.DEFAULT_LOCATION,
    work_type: Optional[str] = None,
    hours_old: int = settings.DEFAULT_HOURS_OLD,
    results_wanted: int = settings.DEFAULT_RESULTS_WANTED,
    sites: Optional[List[str]] = None,
    db_url: Optional[str] = None
) -> Dict[str, Any]:
    is_remote = bool(work_type and "remote" in work_type.lower())
    job_type = work_type.lower() if work_type and work_type.lower() in ["fulltime", "parttime", "contract", "internship"] else None

    if scrape_jobs != getattr(ResilientScraperService, "scrape_jobs", None):
        import services.scraper_engine as se
        orig = se.scrape_jobs
        try:
            se.scrape_jobs = scrape_jobs
            jobs = ResilientScraperService.execute_search(
                search_term=search_term,
                location=location,
                results_wanted=results_wanted,
                hours_old=hours_old,
                is_remote=is_remote,
                sites=sites,
                job_type=job_type,
                db_url=db_url
            )
        finally:
            se.scrape_jobs = orig
    else:
        jobs = ResilientScraperService.execute_search(
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            is_remote=is_remote,
            sites=sites,
            job_type=job_type,
            db_url=db_url
        )

    df = pd.DataFrame(jobs) if jobs else pd.DataFrame()
    return {
        "total_scraped": len(jobs),
        "inserted": len(jobs),
        "skipped": 0,
        "jobs": jobs,
        "dataframe": df
    }
