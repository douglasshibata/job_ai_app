import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

try:
    from job_ai_app.config import DATABASE_URL
except ImportError:
    from config import DATABASE_URL

Base = declarative_base()


class JobPosting(Base):
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_url = Column(String(1024), unique=True, nullable=True, index=True)
    raw_hash = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(255), nullable=False)
    company = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    site = Column(String(100), nullable=True)
    job_type = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    date_posted = Column(String(100), nullable=True)
    min_amount = Column(Float, nullable=True)
    max_amount = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)
    is_remote = Column(Boolean, default=False)

    # Fit Evaluation Fields
    match_score = Column(Integer, default=0, index=True)
    matching_skills = Column(Text, default="[]")  # Stored as JSON list string
    missing_skills = Column(Text, default="[]")   # Stored as JSON list string
    verdict = Column(String(50), default="UNEVALUATED")
    fit_summary = Column(Text, nullable=True)

    # Application Lifecycle Status
    status = Column(String(50), default="Saved")  # Saved, Applied, Interviewing, Rejected, Offer

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationship to ApplicationKit
    application_kits = relationship("ApplicationKit", back_populates="job", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_url": self.job_url,
            "raw_hash": self.raw_hash,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "site": self.site,
            "job_type": self.job_type,
            "description": self.description,
            "date_posted": self.date_posted,
            "min_amount": self.min_amount,
            "max_amount": self.max_amount,
            "currency": self.currency,
            "is_remote": self.is_remote,
            "match_score": self.match_score,
            "matching_skills": json.loads(self.matching_skills or "[]"),
            "missing_skills": json.loads(self.missing_skills or "[]"),
            "verdict": self.verdict,
            "fit_summary": self.fit_summary,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ApplicationKit(Base):
    __tablename__ = "application_kits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("job_postings.id"), nullable=False, index=True)
    tailored_cover_letter = Column(Text, nullable=False)
    customized_bullets = Column(Text, nullable=False)  # JSON array string or text
    audit_status = Column(String(50), default="Passed")  # Passed, Needs Review, Warning
    audit_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    job = relationship("JobPosting", back_populates="application_kits")


def compute_job_hash(title: str, company: str, location: str, job_url: Optional[str] = None) -> str:
    """Generate a unique SHA256 hash for job deduplication."""
    if job_url and str(job_url).strip():
        base = str(job_url).strip().lower()
    else:
        norm_title = (title or "").strip().lower()
        norm_company = (company or "").strip().lower()
        norm_location = (location or "").strip().lower()
        base = f"{norm_title}|{norm_company}|{norm_location}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# Global engine and SessionMaker singleton
_engine = None
_SessionLocal = None


def get_engine(db_url: Optional[str] = None):
    global _engine, _SessionLocal
    target_url = db_url or DATABASE_URL
    if _engine is None or (db_url is not None and str(_engine.url) != target_url):
        _engine = create_engine(
            target_url,
            connect_args={"check_same_thread": False} if target_url.startswith("sqlite") else {}
        )
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def init_db(db_url: Optional[str] = None):
    engine = get_engine(db_url)
    Base.metadata.create_all(bind=engine)


def get_db_session(db_url: Optional[str] = None) -> Session:
    get_engine(db_url)
    return _SessionLocal()


def save_job_postings(jobs_data: List[Dict[str, Any]], db_url: Optional[str] = None) -> Tuple[int, int]:
    """
    Saves a list of job dictionary records into the database with strict deduplication.
    Returns a tuple of (inserted_count, skipped_count).
    """
    init_db(db_url)
    session = get_db_session(db_url)
    inserted = 0
    skipped = 0

    try:
        existing_hashes = set(r[0] for r in session.query(JobPosting.raw_hash).all())
        existing_urls = set(r[0] for r in session.query(JobPosting.job_url).filter(JobPosting.job_url.isnot(None)).all())

        new_objects = []
        for item in jobs_data:
            job_url = str(item.get("job_url", "")).strip() if item.get("job_url") else None
            title = str(item.get("title", "Unknown")).strip()
            company = str(item.get("company", "Unknown")).strip()
            location = str(item.get("location", "")).strip()

            raw_hash = item.get("raw_hash") or compute_job_hash(title, company, location, job_url)

            if raw_hash in existing_hashes or (job_url and job_url in existing_urls):
                skipped += 1
                continue

            # Process skill lists if present
            matching_skills = item.get("matching_skills", [])
            if isinstance(matching_skills, list):
                matching_skills_str = json.dumps(matching_skills)
            else:
                matching_skills_str = str(matching_skills or "[]")

            missing_skills = item.get("missing_skills", [])
            if isinstance(missing_skills, list):
                missing_skills_str = json.dumps(missing_skills)
            else:
                missing_skills_str = str(missing_skills or "[]")

            posting = JobPosting(
                job_url=job_url,
                raw_hash=raw_hash,
                title=title,
                company=company,
                location=location,
                site=str(item.get("site", "")),
                job_type=str(item.get("job_type", "")),
                description=str(item.get("description", "")),
                date_posted=str(item.get("date_posted", "")),
                min_amount=float(item["min_amount"]) if item.get("min_amount") is not None and str(item.get("min_amount")).replace('.', '', 1).isdigit() else None,
                max_amount=float(item["max_amount"]) if item.get("max_amount") is not None and str(item.get("max_amount")).replace('.', '', 1).isdigit() else None,
                currency=str(item.get("currency", "USD")),
                is_remote=bool(item.get("is_remote", False)),
                match_score=int(item.get("match_score", 0)),
                matching_skills=matching_skills_str,
                missing_skills=missing_skills_str,
                verdict=str(item.get("verdict", "UNEVALUATED")),
                fit_summary=str(item.get("fit_summary", "")) if item.get("fit_summary") else None,
                status=str(item.get("status", "Saved"))
            )

            new_objects.append(posting)
            existing_hashes.add(raw_hash)
            if job_url:
                existing_urls.add(job_url)
            inserted += 1

        if new_objects:
            session.add_all(new_objects)
            session.commit()

        return inserted, skipped
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_jobs(
    min_match_score: int = 0,
    search_query: Optional[str] = None,
    verdict_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    db_url: Optional[str] = None
) -> List[JobPosting]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        query = session.query(JobPosting).filter(JobPosting.match_score >= min_match_score)

        if search_query and search_query.strip():
            sq = f"%{search_query.strip()}%"
            query = query.filter(
                (JobPosting.title.ilike(sq)) |
                (JobPosting.company.ilike(sq)) |
                (JobPosting.location.ilike(sq)) |
                (JobPosting.description.ilike(sq))
            )

        if verdict_filter and verdict_filter != "ALL":
            query = query.filter(JobPosting.verdict == verdict_filter)

        if status_filter and status_filter != "ALL":
            query = query.filter(JobPosting.status == status_filter)

        query = query.order_by(JobPosting.match_score.desc(), JobPosting.id.desc())
        return query.all()
    finally:
        session.close()


def get_job_by_id(job_id: int, db_url: Optional[str] = None) -> Optional[JobPosting]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        return session.query(JobPosting).filter(JobPosting.id == job_id).first()
    finally:
        session.close()


def update_job_fit(job_id: int, fit_report: Dict[str, Any], db_url: Optional[str] = None) -> Optional[JobPosting]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        job = session.query(JobPosting).filter(JobPosting.id == job_id).first()
        if not job:
            return None

        job.match_score = int(fit_report.get("match_score", 0))
        matching_skills = fit_report.get("matching_skills", [])
        job.matching_skills = json.dumps(matching_skills) if isinstance(matching_skills, list) else str(matching_skills)

        missing_skills = fit_report.get("missing_skills", [])
        job.missing_skills = json.dumps(missing_skills) if isinstance(missing_skills, list) else str(missing_skills)

        job.verdict = str(fit_report.get("verdict", "UNEVALUATED"))
        job.fit_summary = str(fit_report.get("fit_summary", ""))
        job.updated_at = datetime.now(timezone.utc)

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
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        bullets_str = json.dumps(customized_bullets) if isinstance(customized_bullets, list) else str(customized_bullets)

        app_kit = session.query(ApplicationKit).filter(ApplicationKit.job_id == job_id).first()
        if app_kit:
            app_kit.tailored_cover_letter = tailored_cover_letter
            app_kit.customized_bullets = bullets_str
            app_kit.audit_status = audit_status
            app_kit.audit_notes = audit_notes
            app_kit.created_at = datetime.now(timezone.utc)
        else:
            app_kit = ApplicationKit(
                job_id=job_id,
                tailored_cover_letter=tailored_cover_letter,
                customized_bullets=bullets_str,
                audit_status=audit_status,
                audit_notes=audit_notes
            )
            session.add(app_kit)

        session.commit()
        session.refresh(app_kit)
        return app_kit
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_application_kit_by_job_id(job_id: int, db_url: Optional[str] = None) -> Optional[ApplicationKit]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        return session.query(ApplicationKit).filter(ApplicationKit.job_id == job_id).first()
    finally:
        session.close()


def update_job_status(job_id: int, new_status: str, db_url: Optional[str] = None) -> Optional[JobPosting]:
    init_db(db_url)
    session = get_db_session(db_url)
    try:
        job = session.query(JobPosting).filter(JobPosting.id == job_id).first()
        if job:
            job.status = new_status
            job.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(job)
        return job
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
