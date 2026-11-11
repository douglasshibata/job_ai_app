"""View: Analytics & Kanban Dashboard for Semantic Matching & Application Lifecycle."""
import streamlit as st
from services.vector_search import VectorSearchService
from core.database import get_job_listings, update_job_status
from ui.components import render_metric_cards, render_kanban_board

def render_analytics_view(master_resume: str):
    st.subheader("📊 Análise Vetorial & Kanban de Candidaturas")

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        min_score = st.slider("Score Mínimo Vetorial", min_value=0, max_value=100, value=0, step=5)
    with fcol2:
        search_kw = st.text_input("Filtrar por Termo", placeholder="ex: Python, Tech Lead...")
    with fcol3:
        status_flt = st.selectbox("Status", ["ALL", "NEW", "SHORTLISTED", "APPLIED", "INTERVIEWING", "REJECTED", "OFFER"])

    all_jobs = get_job_listings(
        min_score=float(min_score),
        search_query=search_kw,
        status_filter=status_flt
    )

    total_jobs = len(all_jobs)
    high_matches = len([j for j in all_jobs if (j.semantic_score or 0) >= 75])
    avg_score = sum(j.semantic_score or 0 for j in all_jobs) / max(total_jobs, 1)
    applied_count = len([j for j in all_jobs if j.status.upper() == "APPLIED"])

    render_metric_cards(total_jobs, high_matches, avg_score, applied_count)

    st.markdown("---")

    view_mode = st.radio("Modo de Visualização:", ["Quadro Kanban", "Lista Detalhada"], horizontal=True)

    if view_mode == "Quadro Kanban":
        render_kanban_board(all_jobs, on_status_change=update_job_status)
    else:
        if not all_jobs:
            st.warning("Nenhuma vaga encontrada com os critérios informados.")
        else:
            for job in all_jobs:
                with st.expander(f"**{job.title}** na **{job.company}** — Score: {job.semantic_score or 0:.1f}% [{job.source_board.upper()}]"):
                    col_l, col_r = st.columns([2, 1])
                    with col_l:
                        st.write(f"**Local:** {job.location or 'Remoto'} | **Board:** {job.source_board}")
                        if job.salary_min or job.salary_max:
                            st.write(f"**Faixa Salarial:** {job.currency} {job.salary_min or 0:,.0f} - {job.salary_max or 0:,.0f}")
                        st.markdown("##### Descrição Sanitizada")
                        st.text_area("Descrição", value=job.description, height=120, key=f"desc_{job.id}")

                    with col_r:
                        st.markdown(f"**Status Atual:** `{job.status}`")
                        new_st = st.selectbox(
                            "Atualizar Status",
                            ["NEW", "SHORTLISTED", "APPLIED", "INTERVIEWING", "REJECTED", "OFFER"],
                            index=["NEW", "SHORTLISTED", "APPLIED", "INTERVIEWING", "REJECTED", "OFFER"].index(job.status) if job.status in ["NEW", "SHORTLISTED", "APPLIED", "INTERVIEWING", "REJECTED", "OFFER"] else 0,
                            key=f"st_sel_{job.id}"
                        )
                        if new_st != job.status:
                            update_job_status(job.id, new_st)
                            st.rerun()

                        if job.job_url:
                            st.link_button("🌐 Abrir Vaga Original", job.job_url)
