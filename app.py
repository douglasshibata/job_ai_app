"""Entrypoint principal da aplicação Streamlit (job_ai_app)."""
import os
import sys
from pathlib import Path
import streamlit as st
from pypdf import PdfReader

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from core.config import settings
from core.database import init_db
from services.llm_provider import UnifiedLLMGateway, GroqProvider, GeminiProvider, OllamaProvider
from services.application_agent import ApplicationAgentPipeline
from ui.views.search_view import render_search_view
from ui.views.analytics_view import render_analytics_view
from ui.views.kit_builder_view import render_kit_builder_view

init_db()

st.set_page_config(
    page_title="AI Job Search & Application Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

def extract_resume_text(uploaded_file, text_input: str) -> str:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                pages_text = [p.extract_text() or "" for p in reader.pages]
                full_text = "\n".join(pages_text).strip()
                if full_text:
                    return full_text
            elif uploaded_file.name.endswith((".txt", ".md")):
                return uploaded_file.getvalue().decode("utf-8").strip()
        except Exception as err:
            st.error(f"Erro ao extrair texto do currículo: {err}")
    return text_input.strip()

st.sidebar.title("⚙️ Controle Operacional")
st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Gateway de Inferência LLM")
provider_options = ["groq", "gemini", "ollama"]
selected_provider = st.sidebar.selectbox(
    "Provedor Primário",
    provider_options,
    index=provider_options.index(settings.PRIMARY_LLM_PROVIDER) if settings.PRIMARY_LLM_PROVIDER in provider_options else 0
)

groq_key = st.sidebar.text_input("Groq API Key", value=settings.GROQ_API_KEY, type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", value=settings.GEMINI_API_KEY, type="password")
ollama_host = st.sidebar.text_input("Ollama Host", value=settings.OLLAMA_HOST)

llm_gateway = UnifiedLLMGateway(primary_provider=selected_provider)
llm_gateway.providers["groq"] = GroqProvider(api_key=groq_key)
llm_gateway.providers["gemini"] = GeminiProvider(api_key=gemini_key)
llm_gateway.providers["ollama"] = OllamaProvider(host=ollama_host)

agent_pipeline = ApplicationAgentPipeline(llm_gateway=llm_gateway)

chain = llm_gateway.get_failover_chain()
active = [name.upper() for name, p in chain if p.is_available()]
if active:
    st.sidebar.success(f"Provedores Ativos: {', '.join(active)}")
else:
    st.sidebar.info("Modo Heurístico Ativo (Sem chaves de API remotas configuradas)")

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Currículo Mestre (Truth Source)")
uploaded_cv = st.sidebar.file_uploader("Upload de Currículo (PDF, TXT, MD)", type=["pdf", "txt", "md"])
cv_text = st.sidebar.text_area("Ou Cole o Texto do Currículo", height=180, placeholder="Cole aqui seu histórico profissional...")
master_resume = extract_resume_text(uploaded_cv, cv_text)

if master_resume:
    st.sidebar.success(f"Currículo Carregado ({len(master_resume)} caracteres)")
else:
    st.sidebar.warning("Carregue seu currículo para calcular aderência e gerar kits.")

st.markdown("<div class='main-title'>💼 AI Job Search & Application Preparation Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Agregação Concorrente de Vagas, Classificação Vetorial (FastEmbed) e Pipeline Anti-Alucinação</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🔍 Scraping & Ingestão",
    "📊 Análise Vetorial & Kanban",
    "📝 Kit de Candidatura (Drafter-Auditor)"
])

with tab1:
    render_search_view(master_resume=master_resume)

with tab2:
    render_analytics_view(master_resume=master_resume)

with tab3:
    render_kit_builder_view(master_resume=master_resume, agent_pipeline=agent_pipeline)
