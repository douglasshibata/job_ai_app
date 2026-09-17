import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

try:
    from jobspy import scrape_jobs
except ImportError:
    scrape_jobs = None

try:
    from job_ai_app.config import DEFAULT_SEARCH_TERM, DEFAULT_LOCATION, DEFAULT_HOURS_OLD, DEFAULT_RESULTS_WANTED, DEFAULT_SITES
    from job_ai_app.database import compute_job_hash, save_job_postings, JobPosting
except ImportError:
    from config import DEFAULT_SEARCH_TERM, DEFAULT_LOCATION, DEFAULT_HOURS_OLD, DEFAULT_RESULTS_WANTED, DEFAULT_SITES
    from database import compute_job_hash, save_job_postings, JobPosting

logger = logging.getLogger(__name__)


def fetch_job_listings(
    search_term: str = DEFAULT_SEARCH_TERM,
    location: str = DEFAULT_LOCATION,
    work_type: Optional[str] = None,
    hours_old: int = DEFAULT_HOURS_OLD,
    results_wanted: int = DEFAULT_RESULTS_WANTED,
    sites: Optional[List[str]] = None,
    db_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes multi-board scraping via JobSpy, cleans & normalizes job data,
    deduplicates against existing DB records, and persists new postings.

    Returns a dict with:
        - "total_scraped": total raw scraped count
        - "inserted": new jobs added to DB
        - "skipped": duplicate jobs skipped
        - "jobs": list of dicts for newly inserted jobs or all scraped jobs
        - "dataframe": normalized pandas DataFrame
    """
    if sites is None:
        sites = DEFAULT_SITES

    is_remote = False
    if work_type and "remote" in work_type.lower():
        is_remote = True

    df = pd.DataFrame()
    if scrape_jobs is not None:
        try:
            # Map work_type if applicable
            job_type_filter = None
            if work_type and work_type.lower() in ["fulltime", "parttime", "contract", "internship"]:
                job_type_filter = work_type.lower()

            logger.info(f"Scraping jobs for '{search_term}' in '{location}' across sites: {sites}")
            df = scrape_jobs(
                site_name=sites,
                search_term=search_term,
                location=location,
                results_wanted=results_wanted,
                hours_old=hours_old,
                is_remote=is_remote,
                job_type=job_type_filter
            )
        except Exception as e:
            logger.error(f"Error during JobSpy scrape_jobs execution: {e}")
            df = pd.DataFrame()

    if df.empty:
        return {
            "total_scraped": 0,
            "inserted": 0,
            "skipped": 0,
            "jobs": [],
            "dataframe": pd.DataFrame()
        }

    # Normalize DataFrame columns
    normalized_records = []
    for _, row in df.iterrows():
        title = str(row.get("title", "Unknown")).strip() if pd.notna(row.get("title")) else "Unknown"
        company = str(row.get("company", "Unknown")).strip() if pd.notna(row.get("company")) else "Unknown"
        job_location = str(row.get("location", location)).strip() if pd.notna(row.get("location")) else location
        job_url = str(row.get("job_url", "")).strip() if pd.notna(row.get("job_url")) else None

        # Site source
        site = str(row.get("site", "unknown")).strip() if pd.notna(row.get("site")) else "unknown"

        # Description
        description = str(row.get("description", "")).strip() if pd.notna(row.get("description")) else ""

        # Date posted
        date_posted = str(row.get("date_posted", "")) if pd.notna(row.get("date_posted")) else ""

        # Salary
        min_amount = float(row["min_amount"]) if pd.notna(row.get("min_amount")) else None
        max_amount = float(row["max_amount"]) if pd.notna(row.get("max_amount")) else None
        currency = str(row.get("currency", "USD")) if pd.notna(row.get("currency")) else "USD"

        # Remote status
        row_remote = bool(row.get("is_remote")) if pd.notna(row.get("is_remote")) else is_remote

        # Job type
        jtype = str(row.get("job_type", work_type or "")) if pd.notna(row.get("job_type")) else (work_type or "")

        raw_hash = compute_job_hash(title, company, job_location, job_url)

        normalized_records.append({
            "job_url": job_url,
            "raw_hash": raw_hash,
            "title": title,
            "company": company,
            "location": job_location,
            "site": site,
            "job_type": jtype,
            "description": description,
            "date_posted": date_posted,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "currency": currency,
            "is_remote": row_remote,
            "match_score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "verdict": "UNEVALUATED",
            "fit_summary": "",
            "status": "Saved"
        })

    # Save to database
    inserted, skipped = save_job_postings(normalized_records, db_url=db_url)

    normalized_df = pd.DataFrame(normalized_records)

    return {
        "total_scraped": len(normalized_records),
        "inserted": inserted,
        "skipped": skipped,
        "jobs": normalized_records,
        "dataframe": normalized_df
    }
