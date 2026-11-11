"""Backward compatibility layer for matcher & agent pipeline."""
from typing import List, Optional
from models.schemas import FitReport, DraftMaterials as ApplicationKitReport, CertifiedAuditReport as AuditReport
from services.llm_provider import UnifiedLLMGateway as UnifiedLLMClient
from services.application_agent import ApplicationAgentPipeline

def heuristic_fallback_fit(job_description: str, master_resume: str) -> FitReport:
    resume_lower = master_resume.lower()
    jd_lower = job_description.lower()
    common_keywords = [
        "python", "java", "golang", "react", "typescript", "javascript", "sql", "postgresql",
        "aws", "docker", "kubernetes", "api", "fastapi", "django", "git", "ci/cd",
        "rest", "microservices", "agile", "spark", "llm", "ai", "testing", "pytest"
    ]
    matched = [kw for kw in common_keywords if kw in resume_lower and kw in jd_lower]
    missing = [kw for kw in common_keywords if kw in jd_lower and kw not in resume_lower]

    total_reqs = max(len(matched) + len(missing), 1)
    score = int((len(matched) / total_reqs) * 100) if matched else 50
    verdict = "STRONG_MATCH" if score >= 75 else ("APPLY" if score >= 50 else "PASS")

    return FitReport(
        match_score=score,
        matching_skills=matched or ["General Experience"],
        missing_skills=missing,
        verdict=verdict,
        fit_summary=f"Heuristic analysis identified {len(matched)} matching skills and {len(missing)} missing keywords."
    )

def evaluate_job_fit(job_description: str, master_resume: str, llm_client: Optional[UnifiedLLMClient] = None) -> FitReport:
    client = llm_client or UnifiedLLMClient()
    prompt = f"Evaluate fit between Candidate Resume:\n{master_resume}\nand Job Description:\n{job_description}"
    return client.generate_structured(prompt, FitReport)

def generate_application_kit(job_title: str, company: str, job_description: str, master_resume: str, llm_client: Optional[UnifiedLLMClient] = None) -> ApplicationKitReport:
    client = llm_client or UnifiedLLMClient()
    prompt = f"Generate application kit for {job_title} at {company}. Resume:\n{master_resume}\nJob:\n{job_description}"
    draft = client.generate_structured(prompt, ApplicationKitReport)
    return draft

def review_application_kit(master_resume: str, job_description: str, cover_letter: str, customized_bullets: List[str], llm_client: Optional[UnifiedLLMClient] = None) -> AuditReport:
    pipeline = ApplicationAgentPipeline(llm_gateway=llm_client)
    report = pipeline.process_application(master_resume=master_resume, job_description=job_description)
    return report
