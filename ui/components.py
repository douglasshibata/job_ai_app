"""Reusable UI components for Streamlit dashboard."""
import streamlit as st
from typing import List, Callable
from models.entities import JobListing

def render_metric_cards(total_jobs: int, high_matches: int, avg_score: float, applied_count: int):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Vagas", total_jobs)
    with col2:
        st.metric("Alta Aderência (>75%)", high_matches)
    with col3:
        st.metric("Score Semântico Médio", f"{avg_score:.1f}%")
    with col4:
        st.metric("Candidaturas Enviadas", applied_count)

def render_badge(score: float) -> str:
    if score >= 75:
        return f"<span style='background-color:#DEF7EC;color:#03543F;padding:4px 10px;border-radius:12px;font-weight:600;'>{score:.1f}% Match</span>"
    elif score >= 50:
        return f"<span style='background-color:#FEF08A;color:#713F12;padding:4px 10px;border-radius:12px;font-weight:600;'>{score:.1f}% Match</span>"
    else:
        return f"<span style='background-color:#FEE2E2;color:#991B1B;padding:4px 10px;border-radius:12px;font-weight:600;'>{score:.1f}% Match</span>"

def render_kanban_board(jobs: List[JobListing], on_status_change: Callable[[int, str], None]):
    columns = ["NEW", "SHORTLISTED", "APPLIED", "INTERVIEWING", "REJECTED", "OFFER"]
    col_objects = st.columns(len(columns))

    for idx, col_name in enumerate(columns):
        with col_objects[idx]:
            st.markdown(f"### {col_name}")
            col_jobs = [j for j in jobs if j.status.upper() == col_name]
            st.caption(f"{len(col_jobs)} vagas")

            for job in col_jobs[:8]:
                with st.container(border=True):
                    st.markdown(f"**{job.title}**")
                    st.caption(f"{job.company} • {job.location or 'Remoto'}")
                    st.markdown(render_badge(job.semantic_score or 0.0), unsafe_allow_html=True)

                    target_status = st.selectbox(
                        "Mover:",
                        columns,
                        index=columns.index(col_name),
                        key=f"kanban_sel_{job.id}"
                    )
                    if target_status != col_name:
                        on_status_change(job.id, target_status)
                        st.rerun()
