"""Unit tests for closed-domain factual auditor & anti-hallucination guardrail."""
import pytest
from services.application_agent import ApplicationAgentPipeline, enforce_strict_grounding
from models.schemas import CertifiedAuditReport

def test_enforce_strict_grounding_rejects_hallucinated_metrics():
    master_resume = """
    Software Engineer with experience in Python and PostgreSQL.
    Maintained internal billing system and reduced database query latency.
    """

    draft_letter = "In my previous role, I achieved a 45% reduction in latency and improved throughput by 10x."
    draft_bullets = [
        "Architected cache cluster resulting in 40% cost savings.",
        "Delivered core features on time with high code coverage."
    ]

    certified_letter, certified_bullets, warnings = enforce_strict_grounding(
        draft_letter=draft_letter,
        draft_bullets=draft_bullets,
        master_resume=master_resume
    )

    assert len(warnings) >= 3
    assert "45%" not in certified_letter
    assert "10x" not in certified_letter
    assert "40%" not in certified_bullets[0]
    assert "Delivered core features on time" in certified_bullets[1]

def test_factual_auditor_pipeline_grounding():
    pipeline = ApplicationAgentPipeline()

    master_resume = """
    Senior Data Engineer. Tech stack: Python, Apache Spark, Airflow, Snowflake.
    Lead engineer for customer churn prediction data pipeline.
    """

    job_description = "Looking for Senior Data Engineer with Python, Spark, and Airflow experience."

    report = pipeline.process_application(
        master_resume=master_resume,
        job_description=job_description
    )

    assert isinstance(report, CertifiedAuditReport)
    assert report.ats_match_score > 0
    assert len(report.certified_bullets) > 0
    assert len(report.certified_cover_letter) > 0
