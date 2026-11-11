"""Backward compatibility layer for database functions."""
import json
from typing import List, Dict, Any, Optional, Tuple
from core.database import (
    get_engine, get_db_session, init_db, compute_job_hash, compute_job_fingerprint,
    save_job_listings_batch, get_job_listings, get_job_listing_by_id,
    update_job_status, save_application_record, get_application_record_by_job_id,
    DATABASE_URL
)
from models.entities import JobListing as JobPosting, ApplicationRecord as ApplicationKit

def save_job_postings(jobs_data: List[Dict[str, Any]], db_url: Optional[str] = None) -> Tuple[int, int]:
    return save_job_listings_batch(jobs_data, db_url=db_url)

def get_jobs(
    min_match_score: int = 0,
    search_query: Optional[str] = None,
    verdict_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    db_url: Optional[str] = None
) -> List[JobPosting]:
    return get_job_listings(
        min_score=float(min_match_score),
        search_query=search_query,
        status_filter=status_filter,
        verdict_filter=verdict_filter,
        db_url=db_url
    )

def get_job_by_id(job_id: int, db_url: Optional[str] = None) -> Optional[JobPosting]:
    return get_job_listing_by_id(job_id, db_url=db_url)

def update_job_fit(job_id: int, fit_report: Dict[str, Any], db_url: Optional[str] = None) -> Optional[JobPosting]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        job = session.get(JobPosting, job_id)
        if not job:
            return None
        job.ats_score = int(fit_report.get("match_score", 0))
        job.semantic_score = float(fit_report.get("match_score", 0))
        job.matching_skills = json.dumps(fit_report.get("matching_skills", []))
        job.missing_skills = json.dumps(fit_report.get("missing_skills", []))
        job.verdict = str(fit_report.get("verdict", "UNEVALUATED"))
        job.fit_summary = str(fit_report.get("fit_summary", ""))
        session.commit()
        session.refresh(job)
        return job
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def save_application_kit(
    job_id: int,
    tailored_cover_letter: str,
    customized_bullets: Any,
    audit_status: str = "Passed",
    audit_notes: str = "",
    db_url: Optional[str] = None
) -> ApplicationKit:
    bullets_list = customized_bullets if isinstance(customized_bullets, list) else [str(customized_bullets)]
    return save_application_record(
        job_id=job_id,
        certified_cover_letter=tailored_cover_letter,
        certified_bullets=bullets_list,
        audit_status=audit_status,
        ats_match_score=85,
        hallucination_warnings=[audit_notes] if audit_notes else [],
        db_url=db_url
    )

def get_application_kit_by_job_id(job_id: int, db_url: Optional[str] = None) -> Optional[ApplicationKit]:
    return get_application_record_by_job_id(job_id, db_url=db_url)
