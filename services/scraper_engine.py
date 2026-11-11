"""Resilient scraping service wrapping JobSpy with random jitter, HTML sanitization, and atomic deduplication."""
import logging
import random
import time
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    from jobspy import scrape_jobs
except ImportError:
    scrape_jobs = None

from core.database import (
    SessionLocal,
    compute_job_fingerprint,
    save_job_listings_batch,
    get_db_session,
    init_db
)
from core.security import sanitize_job_description
from models.entities import JobListing

logger = logging.getLogger(__name__)

class ResilientScraperService:
    SUPPORTED_SITES = ["linkedin", "indeed", "glassdoor", "zip_recruiter"]

    @classmethod
    def apply_jitter(cls, min_delay: float = 2.5, max_delay: float = 7.0):
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)

    @classmethod
    def execute_search(
        cls,
        search_term: str,
        location: str,
        results_wanted: int = 20,
        hours_old: int = 72,
        is_remote: bool = False,
        sites: Optional[List[str]] = None,
        job_type: Optional[str] = None,
        db_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        target_sites = sites or cls.SUPPORTED_SITES
        raw_jobs = pd.DataFrame()

        scraper_func = getattr(cls, "scrape_jobs", None) or scrape_jobs

        if scraper_func is not None:
            accumulated_dfs = []
            for site in target_sites:
                try:
                    cls.apply_jitter(min_delay=0.1, max_delay=0.3)
                    site_df = scraper_func(
                        site_name=[site],
                        search_term=search_term,
                        location=location,
                        results_wanted=results_wanted,
                        hours_old=hours_old,
                        is_remote=is_remote,
                        job_type=job_type,
                        country_indeed="worldwide"
                    )
                    if isinstance(site_df, pd.DataFrame) and not site_df.empty:
                        accumulated_dfs.append(site_df)
                except Exception as board_err:
                    logger.warning(f"Board error for {site}: {board_err}")

            if accumulated_dfs:
                raw_jobs = pd.concat(accumulated_dfs, ignore_index=True)

        if raw_jobs.empty:
            return []

        persisted_jobs = []
        init_db(db_url)
        db = get_db_session(db_url)

        try:
            seen_fps_in_batch = set(r[0] for r in db.query(JobListing.fingerprint).all())
            for _, row in raw_jobs.iterrows():
                title = str(row.get("title", "")).strip() if pd.notna(row.get("title")) else ""
                company = str(row.get("company", "")).strip() if pd.notna(row.get("company")) else ""
                loc = str(row.get("location", "")).strip() if pd.notna(row.get("location")) else location
                raw_desc = str(row.get("description", "")).strip() if pd.notna(row.get("description")) else ""
                url = str(row.get("job_url", "")).strip() if pd.notna(row.get("job_url")) else ""
                site = str(row.get("site", "unknown")).strip() if pd.notna(row.get("site")) else "unknown"

                if not title or not company:
                    continue

                desc = sanitize_job_description(raw_desc)
                fp = compute_job_fingerprint(title, company, loc)

                if fp in seen_fps_in_batch:
                    continue

                seen_fps_in_batch.add(fp)

                exists = db.query(JobListing).filter(JobListing.fingerprint == fp).first()
                if not exists:
                    min_amt = float(row.get("min_amount")) if pd.notna(row.get("min_amount")) else None
                    max_amt = float(row.get("max_amount")) if pd.notna(row.get("max_amount")) else None
                    curr = str(row.get("currency", "USD")) if pd.notna(row.get("currency")) else "USD"
                    row_remote = bool(row.get("is_remote")) if pd.notna(row.get("is_remote")) else is_remote
                    j_type = str(row.get("job_type", job_type or "")) if pd.notna(row.get("job_type")) else (job_type or "")

                    job_record = JobListing(
                        fingerprint=fp,
                        title=title,
                        company=company,
                        location=loc,
                        job_type=j_type,
                        description=desc,
                        job_url=url,
                        source_board=site,
                        salary_min=min_amt,
                        salary_max=max_amt,
                        currency=curr,
                        is_remote=row_remote,
                        semantic_score=0.0,
                        status="NEW"
                    )
                    db.add(job_record)
                    persisted_jobs.append({
                        "id": None,
                        "fingerprint": fp,
                        "title": title,
                        "company": company,
                        "location": loc,
                        "url": url,
                        "source": site,
                        "description": desc,
                        "salary_min": min_amt,
                        "salary_max": max_amt
                    })
            db.commit()
        except Exception as err:
            db.rollback()
            raise err
        finally:
            db.close()

        return persisted_jobs
