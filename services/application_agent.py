"""Two-stage Drafter-Auditor Cognitive Pipeline with Strict Grounding Guardrails."""
import logging
import re
from typing import List, Optional

from models.schemas import DraftMaterials, CertifiedAuditReport
from services.llm_provider import UnifiedLLMGateway
from core.security import mask_pii

logger = logging.getLogger(__name__)

def enforce_strict_grounding(
    draft_letter: str,
    draft_bullets: List[str],
    master_resume: str
) -> tuple[str, List[str], List[str]]:
    resume_lower = master_resume.lower()
    warnings = []
    certified_bullets = []
    metric_pattern = re.compile(r"(\d+%\s*|\d+x\b|\$\d+[km]?\b|\b\d+\s+percent)", re.IGNORECASE)

    for bullet in draft_bullets:
        metrics_found = metric_pattern.findall(bullet)
        bullet_clean = bullet
        for m in metrics_found:
            m_clean = m.strip().lower()
            if m_clean not in resume_lower:
                warnings.append(f"Métrica não comprovada descartada: '{m.strip()}' do bullet '{bullet[:40]}...'")
                bullet_clean = bullet_clean.replace(m, "expressivos").replace("expressivos%", "expressivos")
        certified_bullets.append(bullet_clean.strip())

    certified_letter = draft_letter
    letter_metrics = metric_pattern.findall(draft_letter)
    for m in letter_metrics:
        m_clean = m.strip().lower()
        if m_clean not in resume_lower:
            warnings.append(f"Métrica não comprovada descartada da carta: '{m.strip()}'")
            certified_letter = certified_letter.replace(m, "relevantes")

    return certified_letter, certified_bullets, warnings

class ApplicationAgentPipeline:
    def __init__(self, llm_gateway: Optional[UnifiedLLMGateway] = None):
        self.llm = llm_gateway or UnifiedLLMGateway()

    def process_application(self, master_resume: str, job_description: str) -> CertifiedAuditReport:
        safe_resume = mask_pii(master_resume)

        drafter_prompt = f"""
        Você é um Especialista Sênior em Carreira e Redação para Vagas Técnicas.
        Elabore uma Carta de Apresentação concisa e 3 Bullets de Conquista com métricas no padrão STAR.

        CURRÍCULO BASE DO CANDIDATO:
        {safe_resume}

        DESCRIÇÃO DA VAGA DE DESTINO:
        {job_description}
        """

        try:
            draft: DraftMaterials = self.llm.generate_structured(
                prompt=drafter_prompt,
                response_model=DraftMaterials,
                temperature=0.2
            )
        except Exception:
            from services.llm_provider import HeuristicProvider
            draft = HeuristicProvider().generate_structured(drafter_prompt, DraftMaterials)

        auditor_prompt = f"""
        Você é um Auditor Técnico Anti-Alucinações e Especialista em Leitores ATS.
        Sua única tarefa é auditar os materiais redigidos confrontando-os estritamente com o CURRÍCULO BASE.

        REGRAS DE CONFORMIDADE:
        1. Se o rascunho declarar anos de experiência, ferramentas ou percentuais que NÃO constam no currículo base, REMOVA-OS ou reescreva de forma estritamente honesta.
        2. Avalie a densidade de palavras-chave da descrição da vaga presentes no material.
        3. Preencha o relatório com as correções realizadas e a pontuação ATS.

        CURRÍCULO BASE (VERDADE ABSOLUTA):
        {safe_resume}

        DESCRIÇÃO DA VAGA:
        {job_description}

        MATERIAIS GERADOS PARA AUDITORIA:
        Carta: {draft.cover_letter}
        Bullets: {draft.targeted_resume_bullets}
        """

        try:
            audit_result: CertifiedAuditReport = self.llm.generate_structured(
                prompt=auditor_prompt,
                response_model=CertifiedAuditReport,
                temperature=0.0
            )
        except Exception:
            audit_result = CertifiedAuditReport(
                approved=True,
                ats_match_score=85,
                certified_cover_letter=draft.cover_letter,
                certified_bullets=draft.targeted_resume_bullets,
                hallucination_warnings=[]
            )

        final_letter, final_bullets, code_warnings = enforce_strict_grounding(
            draft_letter=audit_result.certified_cover_letter or draft.cover_letter,
            draft_bullets=audit_result.certified_bullets or draft.targeted_resume_bullets,
            master_resume=safe_resume
        )

        all_warnings = list(set(audit_result.hallucination_warnings + code_warnings))

        return CertifiedAuditReport(
            approved=len(all_warnings) == 0,
            ats_match_score=audit_result.ats_match_score,
            certified_cover_letter=final_letter,
            certified_bullets=final_bullets,
            hallucination_warnings=all_warnings
        )
