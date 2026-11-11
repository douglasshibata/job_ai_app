"""SQLAlchemy 2.0 Database session management, initialization, and CRUD operations."""
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy import create_engine, select, or_
from sqlalchemy.orm import sessionmaker, Session

from core.config import settings
from models.entities import Base, JobListing, ApplicationRecord

logger = logging.getLogger(__name__)

DATABASE_URL = settings.DATABASE_URL

_engine = None
_SessionLocal = None

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_engine(db_url: Optional[str] = None):
    global _engine, _SessionLocal
    target_url = db_url or DATABASE_URL
    if db_url is None:
        return engine
    if _engine is None or str(_engine.url) != target_url:
        connect_args = {"check_same_thread": False} if target_url.startswith("sqlite") else {}
        _engine = create_engine(target_url, echo=False, connect_args=connect_args)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine

def get_db_session(db_url: Optional[str] = None) -> Session:
    if db_url is None:
        return SessionLocal()
    eng = get_engine(db_url)
    return _SessionLocal()

def init_db(db_url: Optional[str] = None):
    eng = get_engine(db_url)
    Base.metadata.create_all(bind=eng)

def compute_job_fingerprint(title: str, company: str, location: str, job_url: Optional[str] = None) -> str:
    raw = f"{str(title or '').strip().lower()}|{str(company or '').strip().lower()}|{str(location or '').strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def compute_job_hash(title: str, company: str, location: str, job_url: Optional[str] = None) -> str:
    if job_url and str(job_url).strip():
        base = str(job_url).strip().lower()
        return hashlib.sha256(base.encode("utf-8")).hexdigest()
    return compute_job_fingerprint(title, company, location)

def save_job_listings_batch(
    jobs_data: List[Dict[str, Any]],
    db_url: Optional[str] = None
) -> Tuple[int, int]:
    init_db(db_url)
    session = get_db_session(db_url)
    inserted = 0
    skipped = 0

    try:
        existing_fps = set(r[0] for r in session.query(JobListing.fingerprint).all())
        new_objects = []

        for item in jobs_data:
            title = str(item.get("title", "")).strip()
            company = str(item.get("company", "")).strip()
            location = str(item.get("location", "")).strip()
            url = str(item.get("job_url", "")).strip()
            site = str(item.get("source_board", item.get("site", "unknown"))).strip()
            desc = str(item.get("description", "")).strip()

            if not title or not company:
                skipped += 1
                continue

            fp = item.get("fingerprint") or compute_job_fingerprint(title, company, location)

            if fp in existing_fps:
                skipped += 1
                continue

            matching_skills = item.get("matching_skills", [])
            matching_skills_str = json.dumps(matching_skills) if isinstance(matching_skills, list) else str(matching_skills or "[]")
            missing_skills = item.get("missing_skills", [])
            missing_skills_str = json.dumps(missing_skills) if isinstance(missing_skills, list) else str(missing_skills or "[]")

            record = JobListing(
                fingerprint=fp,
                title=title,
                company=company,
                location=location,
                job_type=str(item.get("job_type", "")),
                description=desc,
                job_url=url,
                source_board=site,
                salary_min=float(item["salary_min"]) if item.get("salary_min") is not None and str(item.get("salary_min")).replace('.', '', 1).isdigit() else (
                    float(item["min_amount"]) if item.get("min_amount") is not None and str(item.get("min_amount")).replace('.', '', 1).isdigit() else None
                ),
                salary_max=float(item["salary_max"]) if item.get("salary_max") is not None and str(item.get("salary_max")).replace('.', '', 1).isdigit() else (
                    float(item["max_amount"]) if item.get("max_amount") is not None and str(item.get("max_amount")).replace('.', '', 1).isdigit() else None
                ),
                currency=str(item.get("currency", "USD")),
                is_remote=bool(item.get("is_remote", False)),
                semantic_score=float(item.get("semantic_score", item.get("match_score", 0))),
                ats_score=int(item.get("ats_score", item.get("match_score", 0))),
                verdict=str(item.get("verdict", "UNEVALUATED")),
                fit_summary=str(item.get("fit_summary", "")),
                matching_skills=matching_skills_str,
                missing_skills=missing_skills_str,
                status=str(item.get("status", "NEW"))
            )
            new_objects.append(record)
            existing_fps.add(fp)
            inserted += 1

        if new_objects:
            session.add_all(new_objects)
            session.commit()

        return inserted, skipped
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving job listings batch: {e}")
        raise e
    finally:
        session.close()

def get_job_listings(
    min_score: float = 0.0,
    search_query: Optional[str] = None,
    status_filter: Optional[str] = None,
    verdict_filter: Optional[str] = None,
    db_url: Optional[str] = None
) -> List[JobListing]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        stmt = select(JobListing)

        if min_score > 0:
            stmt = stmt.where(
                or_(
                    JobListing.semantic_score >= min_score,
                    JobListing.ats_score >= int(min_score)
                )
            )

        if search_query and search_query.strip():
            sq = f"%{search_query.strip()}%"
            stmt = stmt.where(
                or_(
                    JobListing.title.ilike(sq),
                    JobListing.company.ilike(sq),
                    JobListing.location.ilike(sq),
                    JobListing.description.ilike(sq)
                )
            )

        if status_filter and status_filter.upper() != "ALL":
            stmt = stmt.where(JobListing.status == status_filter)

        if verdict_filter and verdict_filter.upper() != "ALL":
            stmt = stmt.where(JobListing.verdict == verdict_filter)

        stmt = stmt.order_by(JobListing.semantic_score.desc().nullslast(), JobListing.id.desc())
        return list(session.scalars(stmt).all())
    finally:
        session.close()

def get_job_listing_by_id(job_id: int, db_url: Optional[str] = None) -> Optional[JobListing]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        return session.get(JobListing, job_id)
    finally:
        session.close()

def update_job_semantic_score(
    job_id: int,
    semantic_score: float,
    embedding: Optional[List[float]] = None,
    db_url: Optional[str] = None
) -> Optional[JobListing]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        job = session.get(JobListing, job_id)
        if job:
            job.semantic_score = semantic_score
            if embedding:
                job.embedding_json = json.dumps(embedding)
            session.commit()
            session.refresh(job)
        return job
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def update_job_status(job_id: int, new_status: str, db_url: Optional[str] = None) -> Optional[JobListing]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        job = session.get(JobListing, job_id)
        if job:
            job.status = new_status
            if new_status.upper() == "APPLIED":
                job.applied_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(job)
        return job
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def save_application_record(
    job_id: int,
    certified_cover_letter: str,
    certified_bullets: List[str],
    audit_status: str = "Passed",
    ats_match_score: int = 0,
    hallucination_warnings: Optional[List[str]] = None,
    db_url: Optional[str] = None
) -> ApplicationRecord:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        bullets_str = json.dumps(certified_bullets)
        warnings_str = json.dumps(hallucination_warnings or [])

        record = session.query(ApplicationRecord).filter(ApplicationRecord.job_id == job_id).first()
        if record:
            record.certified_cover_letter = certified_cover_letter
            record.certified_bullets = bullets_str
            record.audit_status = audit_status
            record.ats_match_score = ats_match_score
            record.hallucination_warnings = warnings_str
            record.created_at = datetime.now(timezone.utc)
        else:
            record = ApplicationRecord(
                job_id=job_id,
                certified_cover_letter=certified_cover_letter,
                certified_bullets=bullets_str,
                audit_status=audit_status,
                ats_match_score=ats_match_score,
                hallucination_warnings=warnings_str
            )
            session.add(record)

        job = session.get(JobListing, job_id)
        if job:
            job.ats_score = ats_match_score

        session.commit()
        session.refresh(record)
        return record
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_application_record_by_job_id(job_id: int, db_url: Optional[str] = None) -> Optional[ApplicationRecord]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        return session.query(ApplicationRecord).filter(ApplicationRecord.job_id == job_id).first()
    finally:
        session.close()
