"""SQLAlchemy 2.0 Declarative Entities for job_ai_app."""
import datetime
from typing import Optional, List
from sqlalchemy import String, Integer, Float, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class JobListing(Base):
    __tablename__ = "job_listings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    job_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    job_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_board: Mapped[str] = mapped_column(String(50), nullable=False)

    salary_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    salary_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(10), default="USD", nullable=True)
    is_remote: Mapped[bool] = mapped_column(Boolean, default=False)
    semantic_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    ats_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    verdict: Mapped[str] = mapped_column(String(50), default="UNEVALUATED")
    fit_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    matching_skills: Mapped[str] = mapped_column(Text, default="[]")
    missing_skills: Mapped[str] = mapped_column(Text, default="[]")

    @property
    def match_score(self) -> int:
        return int(self.ats_score or self.semantic_score or 0)

    @match_score.setter
    def match_score(self, value: int):
        self.ats_score = int(value)
        self.semantic_score = float(value)

    @property
    def site(self) -> str:
        return self.source_board

    @site.setter
    def site(self, value: str):
        self.source_board = value

    @property
    def min_amount(self) -> Optional[float]:
        return self.salary_min

    @property
    def max_amount(self) -> Optional[float]:
        return self.salary_max

    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(50), default="NEW", index=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    applied_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    application_records: Mapped[List["ApplicationRecord"]] = relationship(
        "ApplicationRecord", back_populates="job", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "job_type": self.job_type,
            "description": self.description,
            "job_url": self.job_url,
            "source_board": self.source_board,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "currency": self.currency,
            "is_remote": self.is_remote,
            "semantic_score": self.semantic_score,
            "ats_score": self.ats_score,
            "verdict": self.verdict,
            "fit_summary": self.fit_summary,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }

class ApplicationRecord(Base):
    __tablename__ = "application_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("job_listings.id"), index=True, nullable=False)
    certified_cover_letter: Mapped[str] = mapped_column(Text, nullable=False)
    certified_bullets: Mapped[str] = mapped_column(Text, nullable=False)
    audit_status: Mapped[str] = mapped_column(String(50), default="Passed")
    ats_match_score: Mapped[int] = mapped_column(Integer, default=0)
    hallucination_warnings: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    job: Mapped["JobListing"] = relationship("JobListing", back_populates="application_records")

    @property
    def tailored_cover_letter(self) -> str:
        return self.certified_cover_letter

    @tailored_cover_letter.setter
    def tailored_cover_letter(self, val: str):
        self.certified_cover_letter = val

    @property
    def customized_bullets(self) -> str:
        return self.certified_bullets

    @customized_bullets.setter
    def customized_bullets(self, val: str):
        self.certified_bullets = val

    @property
    def audit_notes(self) -> str:
        return self.hallucination_warnings

    @audit_notes.setter
    def audit_notes(self, val: str):
        self.hallucination_warnings = val
