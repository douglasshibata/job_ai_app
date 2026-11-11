"""View: Application Kit Builder powered by Drafter-Auditor Cognitive Pipeline."""
import streamlit as st
import json
from services.application_agent import ApplicationAgentPipeline
from core.database import (
    get_job_listings,
    get_job_listing_by_id,
    save_application_record,
    get_application_record_by_job_id,
    update_job_status
)

def render_kit_builder_view(master_resume: str, agent_pipeline: ApplicationAgentPipeline):
    st.subheader("📝 Kit de Candidatura Certificado (Drafter-Auditor)")
    st.write("Gere cartas de apresentação sob medida e pontos de conquista STAR com auditoria factual estrita contra alucinações.")

    all_jobs = get_job_listings()
    if not all_jobs:
        st.info("Nenhuma vaga cadastrada. Realize uma busca no primeiro painel.")
        return

    job_options = {f"[{j.semantic_score or 0:.0f}% Match] {j.title} - {j.company} (ID: {j.id})": j.id for j in all_jobs}
    selected_label = st.selectbox("Selecione a Vaga de Destino", list(job_options.keys()))
    selected_job_id = job_options[selected_label]
    target_job = get_job_listing_by_id(selected_job_id)

    if not target_job:
        st.error("Vaga não encontrada.")
        return

    st.markdown(f"### Vaga: **{target_job.title}** na **{target_job.company}**")

    if st.button("✨ Gerar / Auditar Kit de Candidatura", type="primary", use_container_width=True):
        if not master_resume:
            st.error("Insira o Currículo Mestre na barra lateral antes de gerar materiais.")
            return

        with st.status("Executando Pipeline Cognitiva Drafter-Auditor...", expanded=True) as status:
            st.write("✍️ Agente 1 (Drafter): Elaborando carta persuasiva e bullets STAR adaptados...")
            st.write("🛡️ Agente 2 (Auditor Factual): Executando auditoria fechada de fatos e checagem ATS...")

            audit_report = agent_pipeline.process_application(
                master_resume=master_resume,
                job_description=target_job.description or f"{target_job.title} at {target_job.company}"
            )

            save_application_record(
                job_id=target_job.id,
                certified_cover_letter=audit_report.certified_cover_letter,
                certified_bullets=audit_report.certified_bullets,
                audit_status="Approved" if audit_report.approved else "Needs Review",
                ats_match_score=audit_report.ats_match_score,
                hallucination_warnings=audit_report.hallucination_warnings
            )

            status.update(label="Kit Gerado e Auditado com Sucesso!", state="complete", expanded=False)
            st.rerun()

    existing_record = get_application_record_by_job_id(target_job.id)
    if existing_record:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Status da Auditoria", existing_record.audit_status)
        with c2:
            st.metric("Compatibilidade ATS", f"{existing_record.ats_match_score}%")
        with c3:
            if st.button("Marcar Vaga como 'APPLIED' 🚀"):
                update_job_status(target_job.id, "APPLIED")
                st.success("Status atualizado para APPLIED!")

        warnings = json.loads(existing_record.hallucination_warnings or "[]")
        if warnings:
            st.warning("⚠️ **Avisos do Auditor Anti-Alucinação (Dados Removidos/Ajustados):**\n" + "\n".join(f"- {w}" for w in warnings))

        st.markdown("---")
        st.subheader("✉️ Carta de Apresentação Certificada")
        editable_letter = st.text_area("Texto da Carta", value=existing_record.certified_cover_letter, height=260)

        st.download_button(
            label="📥 Baixar Carta (.txt)",
            data=editable_letter,
            file_name=f"Carta_{target_job.company.replace(' ', '_')}.txt",
            mime="text/plain"
        )

        st.markdown("---")
        st.subheader("🎯 Bullets de Conquista Auditados (STAR)")
        bullets = json.loads(existing_record.certified_bullets or "[]")
        for b in bullets:
            st.markdown(f"- {b}")

        st.download_button(
            label="📥 Baixar Bullets (.txt)",
            data="\n".join(f"- {b}" for b in bullets),
            file_name=f"Bullets_{target_job.company.replace(' ', '_')}.txt",
            mime="text/plain"
        )
