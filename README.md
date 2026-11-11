# 💼 AI Job Search & Application Preparation Platform (`job_ai_app`)

Plataforma inteligente e desacoplada para agregação concorrente de vagas, classificação vetorial de relevância (*Vector Match Scoring* via FastEmbed), auditoria factual anti-alucinação e geração customizada de kits de candidatura para aprovação em sistemas ATS (*Applicant Tracking Systems*).

---

## 📋 Sumário
- [1. Topologia & Arquitetura de Alto Nível](#1-topologia--arquitetura-de-alto-nível)
- [2. Matriz de Diferenciais Técnicos](#2-matriz-de-diferenciais-técnicos)
- [3. Estrutura do Repositório](#3-estrutura-do-repositório)
- [4. Especificação dos Componentes e Implementação](#4-especificação-dos-componentes-e-implementação)
  - [4.1. Camada de Dados e Migrações (SQLAlchemy + SQLite-Vec)](#41-camada-de-dados-e-migrações-sqlalchemy--sqlite-vec)
  - [4.2. Scraping Concorrente com Resiliência (JobSpy Wrapper)](#42-scraping-concorrente-com-resiliência-jobspy-wrapper)
  - [4.3. Pipeline Agente Drafter-Auditor com Grounding Estrito](#43-pipeline-agente-drafter-auditor-com-grounding-estrito)
- [5. Guia de Configuração e Execução](#5-guia-de-configuração-e-execução)
- [6. Estratégias de Produção & Anti-Bot](#6-estratégias-de-produção--anti-bot)
- [7. Suíte de Testes Automatizados](#7-suíte-de-testes-automatizados)

---

## 1. Topologia & Arquitetura de Alto Nível

A plataforma foi arquitetada em camadas assíncronas e desacopladas. O Streamlit opera estritamente como frontend leve de visualização, comunicando-se com serviços que manipulam threads de scraping, indexação vetorial e inferência de LLM com fallback em cascata (**Groq $\to$ Gemini $\to$ Ollama $\to$ Heurística Local**).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CAMADA DE FONTES & INGESTÃO                           │
│     LinkedIn API/Web  │  Indeed Scraping  │  Glassdoor Core  │  ZipRecruiter    │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ (Async Task Runner / Thread Pool)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       MOTOR DE SCRAPING RESILIENTE                              │
│         - Sanitização de HTML com Bleach & Trafilatura                          │
│         - Normalização Salarial & Deduplicação por Fingerprint SHA-256          │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING & PERSISTÊNCIA MULTIDIMENSIONAL                    │
│   ┌─────────────────────────────────────┐ ┌───────────────────────────────────┐ │
│   │   SQLite Relacional (ACID Local)    │ │   SQLite-Vec / FastEmbed Local    │ │
│   │   - JobPosting & ApplicationTrack   │ │   - Embeddings de Vagas (384-dim) │ │
│   └─────────────────────────────────────┘ └───────────────────────────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  PIPELINE COGNITIVO DRAFTER-AUDITOR (ANTI-ALUCINAÇÃO)           │
│                                                                                 │
│   [Currículo Mestre] ──► [Extração Textual] ──► [Grafo de Competências Fatuais] │
│                                                          │                      │
│   ┌──────────────────────────────────────────────┐       ▼                      │
│   │ Agente 1: Drafter                            │ ──► [Rascunho de Carta &     │
│   │ - Alinhamento Semântico com a Vaga           │     Bullets Customizados]    │
│   └──────────────────────────────────────────────┘       │                      │
│                                                          ▼                      │
│   ┌──────────────────────────────────────────────┐ ◄─────┘                      │
│   │ Agente 2: Auditor Factual (Strict Guardrail) │                              │
│   │ - Rejeita Métricas Não Declaradas no CV      │ ──► [Kit Certificado ATS]    │
│   │ - Cálculo de Densidade de Palavras-Chave     │                              │
│   └──────────────────────────────────────────────┘                              │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD & TRACKING INTERATIVO                     │
│    Filtros Reativos │ Kanban de Vagas │ Editor de Documentos │ Exportação TXT   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Matriz de Diferenciais Técnicos

| Dimensão | Abordagem Convencional | Arquitetura job_ai_app Otimizada |
|---|---|---|
| **Execução de Scraping** | Bloqueia a interface do usuário durante requisições | Execução não-bloqueante em background com cache e checkpoints em disco |
| **Matching de Vagas** | Envio de todo o texto de 50 vagas para LLMs caras | Filtro em dois estágios: Triagem vetorial rápida (embeddings FastEmbed) + Análise profunda apenas no Top 15% |
| **Geração de Currículo/Carta** | Geração criativa livre (alto risco de alucinar métricas/empresas) | Auditoria de Fatos Fechada (*Closed-Domain Fact Checking*): Nenhuma métrica ou percentual é gerado sem lastro no currículo |
| **Tolerância a Falhas de API** | Falha geral se a cota do provedor esgotar | Failover transparente em 4 níveis: Groq (LPU) $\to$ Gemini $\to$ Ollama Local $\to$ Heurística Local |
| **Resiliência Anti-Scraping** | Falha direta sob status HTTP 429/403 | Tratamento com expurgo seletivo de headers, jitter adaptativo (2.5s - 7.0s) e isolamento de falha por board |

---

## 3. Estrutura do Repositório

```
job_ai_app/
├── core/
│   ├── __init__.py
│   ├── config.py              # Pydantic BaseSettings e carregamento de variáveis de ambiente
│   ├── database.py            # Engine SQLAlchemy 2.0 com suporte a SQLite-Vec
│   ├── security.py            # Sanitização de entradas, expurgo de HTML e mascaramento de PII
│   └── exceptions.py          # Exceções tipadas do domínio (ScrapingError, LLMQuotaError)
├── services/
│   ├── __init__.py
│   ├── scraper_engine.py      # Orquestrador do JobSpy com retry exponencial e validação
│   ├── vector_search.py       # Vetorização e cálculo de cosine similarity via FastEmbed
│   ├── llm_provider.py        # Gateway unificado com fallback automático e circuit-breaker
│   └── application_agent.py   # Implementação do ciclo Drafter-Auditor com Pydantic
├── models/
│   ├── __init__.py
│   ├── entities.py            # Tabelas: JobListing, CandidateProfile, ApplicationRecord
│   └── schemas.py             # Schemas Pydantic: FitReport, CoverLetterAudit, ATSScore
├── ui/
│   ├── __init__.py
│   ├── components.py          # Renderizadores de componentes Streamlit, cards e kanban
│   └── views/
│       ├── search_view.py     # Interface de busca e painel de scraping
│       ├── analytics_view.py  # Análise gráfica de gaps de competências e Kanban
│       └── kit_builder_view.py# Geração, auditoria e download do kit de candidatura
├── tests/
│   ├── test_scraper.py        # Mocks de scraping e tolerância a falhas
│   ├── test_fact_auditor.py   # Testes de regressão contra alucinações de LLM
│   ├── test_vector_engine.py  # Testes de alinhamento por similaridade de cossenos
│   └── test_job_ai_app.py     # Suíte de regressão de compatibilidade
├── app.py                     # Entrypoint de inicialização da interface Streamlit
├── requirements.txt           # Dependências com travas de versão estáveis
├── .env.example
└── README.md
```

---

## 4. Especificação dos Componentes e Implementação

### 4.1. Camada de Dados e Migrações (SQLAlchemy + SQLite-Vec)

Definição declarativa em `models/entities.py` e engine em `core/database.py`:
- `JobListing`: Tabela principal contendo hash determinístico SHA-256 (`fingerprint`), dados descritivos, scores semânticos e status Kanban.
- `ApplicationRecord`: Histórico auditado contendo carta de apresentação certificada, bullets STAR aprovados e avisos anti-alucinação.

### 4.2. Scraping Concorrente com Resiliência (JobSpy Wrapper)

Implementado em `services/scraper_engine.py`:
- **Jitter Dinâmico**: Intervalos aleatórios mitigando bloqueios Cloudflare.
- **Isolamento de Falha**: Se um board responder HTTP 403/429, os demais continuam normalmente.
- **Sanitização com Trafilatura**: HTML é convertido para texto enxuto e sem tags indesejadas antes da persistência.

### 4.3. Pipeline Agente Drafter-Auditor com Grounding Estrito

Implementado em `services/application_agent.py`:
- **Drafter**: Gera materiais direcionados à descrição da vaga com foco em clareza e impacto STAR.
- **Auditor Factual (Closed-Domain Guardrail)**: Compara todas as métricas, percentuais e marcos com o currículo original. Qualquer métrica quantitativa não comprovada é descartada e substituída por formulação honesta.

---

## 5. Guia de Configuração e Execução

### 1. Dependências de Sistema e Pacotes Python

Instale em um ambiente virtual isolado:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuração de Variáveis de Ambiente (`.env`)

Crie o arquivo `.env` a partir de `.env.example`:
```bash
cp .env.example .env
```

```env
# Gateway de LLMs com Fallback
PRIMARY_LLM_PROVIDER=groq
GROQ_API_KEY=gsk_exemplo_chave_aqui
GEMINI_API_KEY=AIzaSy_exemplo_chave_aqui
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3

# Armazenamento & SQLite
DATABASE_URL=sqlite:///job_ai_platform.db

# Parâmetros de Scraping
SCRAPER_WORKERS=4
DEFAULT_HOURS_OLD=72
```

### 3. Inicialização do Banco e Servidor Web

```bash
streamlit run app.py
```

---

## 6. Estratégias de Produção & Anti-Bot

Ao escalar buscas no LinkedIn, Indeed, Glassdoor e ZipRecruiter:
- **Jitter & Delays Aleatórios**: Pausas randômicas de 2.5s a 7.0s mitigam bloqueios por IP.
- **Desacoplamento de Sessão**: Interface reativa em `st.status` e `st.session_state` impede bloqueio de WebSocket do Streamlit.
- **Limpeza Prévia com Trafilatura & Bleach**: Reduz o volume de tokens enviados a LLMs em até 65%.

---

## 7. Suíte de Testes Automatizados

Execute os testes com `pytest`:
```bash
python3 -m pytest tests/ -v
```
