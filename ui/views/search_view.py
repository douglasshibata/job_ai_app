"""View: Search & Scrape Interface with background non-blocking execution."""
import streamlit as st
import pandas as pd
from services.scraper_engine import ResilientScraperService
from services.vector_search import VectorSearchService
from core.database import get_job_listings
from core.config import settings

def render_search_view(master_resume: str):
    st.subheader("🔍 Motor de Scraping Concorrente & Resiliente")
    st.markdown("Busca simultânea nos portais **LinkedIn, Indeed, Glassdoor e ZipRecruiter** com jitter anti-bot e sanitização.")

    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Cargo / Palavras-chave", value=settings.DEFAULT_SEARCH_TERM)
        location = st.text_input("Localização", value=settings.DEFAULT_LOCATION)
        is_remote = st.checkbox("Apenas Vagas Remotas", value=True)

    with col2:
        hours_old_map = {"Últimas 24h": 24, "Últimos 3 dias": 72, "Última semana": 168}
        hours_selected = st.selectbox("Período de Publicação", list(hours_old_map.keys()), index=1)
        hours_old = hours_old_map[hours_selected]
        results_wanted = st.slider("Resultados por Board", min_value=5, max_value=50, value=settings.DEFAULT_RESULTS_WANTED, step=5)
        selected_sites = st.multiselect(
            "Boards Selecionados",
            ["linkedin", "indeed", "glassdoor", "zip_recruiter"],
            default=settings.DEFAULT_SITES
        )

    auto_vectorize = st.checkbox("Executar classificação vetorial (FastEmbed) automaticamente após scraping", value=True)

    if st.button("🚀 Iniciar Scraping em Segundo Plano", type="primary", use_container_width=True):
        if not selected_sites:
            st.error("Selecione pelo menos um quadro de vagas para a busca.")
            return

        with st.status("Processando scraping concorrente com contingência...", expanded=True) as status:
            st.write("🛰️ Disparando requisições assíncronas com jitter adaptativo...")
            new_jobs = ResilientScraperService.execute_search(
                search_term=search_term,
                location=location,
                results_wanted=results_wanted,
                hours_old=hours_old,
                is_remote=is_remote,
                sites=selected_sites
            )

            st.write(f"📥 {len(new_jobs)} novas vagas capturadas, sanitizadas e persistidas.")

            if auto_vectorize and master_resume and new_jobs:
                st.write("🧠 Vetorizando descrições e calculando cosine similarity contra Currículo Mestre...")
                all_current_jobs = get_job_listings()
                VectorSearchService.rank_jobs_by_resume(master_resume, all_current_jobs)
                st.write("✅ Classificação vetorial concluída!")

            status.update(label=f"Concluído! {len(new_jobs)} vagas adicionadas.", state="complete", expanded=False)

        st.rerun()

    st.markdown("---")
    st.subheader("📋 Vagas Armazenadas no Banco SQLite")
    stored_jobs = get_job_listings()

    if stored_jobs:
        st.write(f"Total de vagas registradas: **{len(stored_jobs)}**")
        df_summary = pd.DataFrame([
            {
                "ID": j.id,
                "Título": j.title,
                "Empresa": j.company,
                "Local": j.location,
                "Board": j.source_board,
                "Score Vetorial": f"{j.semantic_score or 0.0:.1f}%",
                "Status": j.status
            } for j in stored_jobs[:15]
        ])
        st.dataframe(df_summary, use_container_width=True)
    else:
        st.info("Nenhuma vaga armazenada ainda. Execute uma pesquisa acima.")
