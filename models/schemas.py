"""Pydantic schemas for data validation, fit scoring, and Drafter-Auditor contracts."""
from typing import List, Optional
from pydantic import BaseModel, Field

class DraftMaterials(BaseModel):
    cover_letter: str = Field(description="Carta de apresentação direcionada à vaga")
    targeted_resume_bullets: List[str] = Field(description="Pontos de impacto no padrão STAR")

    @property
    def tailored_cover_letter(self) -> str:
        return self.cover_letter

    @property
    def customized_bullets(self) -> List[str]:
        return self.targeted_resume_bullets

class ApplicationKitReport(DraftMaterials):
    pass

class CertifiedAuditReport(BaseModel):
    approved: bool = Field(description="True se o material não contém alucinações fáticas")
    ats_match_score: int = Field(ge=0, le=100, description="Score de compatibilidade semântica com o edital da vaga")
    certified_cover_letter: str = Field(description="Texto final após remoção de alegações não comprovadas")
    certified_bullets: List[str] = Field(description="Bullets finais validados")
    hallucination_warnings: List[str] = Field(default_factory=list, description="Lista de dados removidos por falta de comprovação")

    @property
    def audit_status(self) -> str:
        return "Passed" if self.approved else "Needs Review"

    @property
    def audit_notes(self) -> str:
        return "; ".join(self.hallucination_warnings) if self.hallucination_warnings else "Passed compliance check."

    @property
    def hallucination_check_passed(self) -> bool:
        return self.approved

    @property
    def ats_score(self) -> int:
        return self.ats_match_score

class AuditReport(CertifiedAuditReport):
    pass

class ATSScore(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    keyword_match_density: float = Field(ge=0.0, le=1.0)
    seniority_alignment: str
    missing_critical_skills: List[str] = Field(default_factory=list)

class FitReport(BaseModel):
    match_score: int = Field(ge=0, le=100, description="Match percentage score from 0 to 100")
    matching_skills: List[str] = Field(description="List of matched technical and soft skills")
    missing_skills: List[str] = Field(description="List of missing skills required by the position")
    verdict: str = Field(description="Decision verdict: APPLY, STRONG_MATCH, or PASS")
    fit_summary: str = Field(description="Concise synthesis explaining alignment and seniority fit")
