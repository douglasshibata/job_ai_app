# 💼 AI Job Search & Application Preparation Platform (`job_ai_app`)

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![JobSpy](https://img.shields.io/badge/JobSpy-Multi--Board-brightgreen.svg)](https://github.com/speedyapply/JobSpy)
[![LLM Free Tier](https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini%20%7C%20Ollama-purple.svg)](#3-integração-com-provedores-llm-gratuitos)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

Plataforma inteligente e automatizada para raspagem multi-board de vagas de emprego, avaliação de aderência (*fit evaluation*) com Inteligência Artificial e geração/auditoria de kits de candidatura (cartas de apresentação sob medida e pontos de conquista otimizados para ATS).

---

## 📋 Sumário
- [1. Visão Geral & Arquitetura](#1-visão-geral--arquitetura)
- [2. Principais Funcionalidades](#2-principais-funcionalidades)
- [3. Integração com Provedores LLM Gratuitos](#3-integração-com-provedores-llm-gratuitos)
- [4. Estrutura do Projeto](#4-estrutura-do-projeto)
- [5. Instalação & Pré-requisitos](#5-instalação--pré-requisitos)
- [6. Configuração de Variáveis de Ambiente (`.env`)](#6-configuração-de-variáveis-de-ambiente-env)
- [7. Execução & Uso do Dashboard Streamlit](#7-execução--uso-do-dashboard-streamlit)
- [8. Suíte de Testes Automatizados](#8-suíte-de-testes-automatizados)
- [9. Segurança & Boas Práticas](#9-segurança--boas-práticas)
- [10. Troubleshooting](#10-troubleshooting)

---

## 1. Visão Geral & Arquitetura

O **`job_ai_app`** combina a biblioteca de scraping concorrente **JobSpy** com a pipeline de avaliação e agentes **Drafter-Reviewer** inspirada no conceito `ai-job-search`. O sistema funciona 100% sem dependências pagas de nuvem, permitindo que qualquer profissional busque oportunidades, calcule pontuações de alinhamento com seu currículo mestre e gere materiais de aplicação de alta conversão.

### Fluxo de Dados & Arquitetura de Componentes

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FONTES DE VAGAS (JOB BOARDS)                          │
│        LinkedIn   │   Indeed   │   Glassdoor   │   ZipRecruiter                 │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   JobSpy Scraper Engine   │
                         │    (job_ai_app/scraper)   │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  Deduplicação & SQLite DB     │
                       │   (SHA256 Hash + job_url)     │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   AI Fit Evaluation Engine    │
                       │     (job_ai_app/matcher)      │
                       └───────────────┬───────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│     Drafter Agent     │                             │    Reviewer Agent     │
│ (Carta de Apresentação│                             │  (Auditoria de ATS e  │
│ e Bullets de Resume)  │                             │  Prevenção de Aluc.)  │
└───────────┬───────────┘                             └───────────┬───────────┘
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │   Painel Web (Streamlit)    │
                        │    (job_ai_app/app.py)      │
                        └─────────────────────────────┘
```

---

## 2. Principais Funcionalidades

1. **Scraping Concorrente Multi-Board**:
   - Captura em tempo real no LinkedIn, Indeed, Glassdoor e ZipRecruiter via `python-jobspy`.
   - Filtros por título/palavras-chave, localização (remoto/híbrido/presencial), limite de data e quantidade por site.

2. **Deduplicação Determinística**:
   - Algoritmo de hash SHA256 combinando `title + company + location` e checagem única de `job_url` em SQLite.

3. **Avaliação Automatizada de Aderência (*Fit Evaluation*)**:
   - Extração do currículo mestre (PDF via `pypdf`, Markdown ou texto plano).
   - Cálculo de score de alinhamento (0–100%), levantamento de competências compatíveis versus faltantes e veredito (`STRONG_MATCH`, `APPLY` ou `PASS`).

4. **Pipeline Agente Drafter-Reviewer**:
   - **Agente Drafter**: Redige cartas de apresentação personalizadas e pontos de conquista focados em resultados quantificáveis.
   - **Agente Reviewer**: Audita os materiais gerados contra o currículo base para eliminar alucinações, assegurar 100% de veracidade e garantir conformidade com leitores ATS.

5. **Acompanhamento do Ciclo de Candidatura**:
   - Gestão de status de aplicação diretamente na interface (*Saved*, *Applied*, *Interviewing*, *Rejected*, *Offer*).

---

## 3. Integração com Provedores LLM Gratuitos

O módulo `llm_client.py` encapsula uma arquitetura de cliente unificado com **failover automático**:

| Provedor | Modelo Padrão | Vantagens |
| :--- | :--- | :--- |
| **Groq API** | `llama-3.3-70b-versatile` / `mixtral-8x7b-32768` | Altíssima velocidade de resposta e parsing JSON estrito. |
| **Google Gemini API** | `gemini-2.5-flash` | Modelo gratuito de última geração com suporte nativo a schemas Pydantic. |
| **Ollama (Local)** | `llama3` / `deepseek-r1` | Execução 100% offline e privada em hardware local. |
| **Heuristic Fallback** | N/A (Algoritmo local) | Mantém o app operacional mesmo sem chaves de API ativas. |

---

## 4. Estrutura do Projeto

```
job_ai_app/
├── config.py             # Configurações globais, chaves de API e URLs do banco
├── database.py           # Modelos SQLAlchemy (JobPosting, ApplicationKit) e CRUD
├── scraper.py            # Wrapper do JobSpy e lógica de deduplicação
├── llm_client.py         # Cliente LLM modular com fallback (Groq/Gemini/Ollama)
├── matcher.py            # Prompts, schemas FitReport/AuditReport e pipeline Drafter-Reviewer
├── app.py                # Dashboard web iterativo em Streamlit
├── requirements.txt      # Dependências do projeto
├── .env.example          # Modelo de arquivo de configuração
├── tests/
│   └── test_job_ai_app.py # Suíte de testes unitários e de integração
└── README.md             # Documentação técnica completa
```

---

## 5. Instalação & Pré-requisitos

### Pré-requisitos
- Python 3.10 ou superior (testado e validado no Python 3.12+).
- Gerenciador de pacotes `pip`.

### Passos de Instalação

```bash
# 1. Navegue até o diretório do projeto
cd job_ai_app

# 2. (Opcional) Crie e ative um ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Linux/macOS
# venv\Scripts\activate   # No Windows

# 3. Instale as dependências requeridas
pip install -r requirements.txt
```

---

## 6. Configuração de Variáveis de Ambiente (`.env`)

Crie um arquivo `.env` na raiz do diretório `job_ai_app/` com base no arquivo de exemplo `.env.example`:

```bash
cp .env.example .env
```

Edite as chaves de API e preferências conforme necessário:

```env
# Chaves de API para Provedores LLM Gratuitos
GROQ_API_KEY=gsk_sua_chave_groq_aqui
GEMINI_API_KEY=AIzaSy_sua_chave_gemini_aqui
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3

# Provedor Principal Padrão (groq, gemini, ou ollama)
PRIMARY_LLM_PROVIDER=groq

# Banco de Dados SQLite
DATABASE_URL=sqlite:///job_ai_app.db

# Parâmetros Padrão de Busca
DEFAULT_SEARCH_TERM=Software Engineer
DEFAULT_LOCATION=Remote
DEFAULT_HOURS_OLD=72
DEFAULT_RESULTS_WANTED=15
```

> **Nota:** Se você não configurar chaves para o Groq ou Gemini, o sistema utilizará o Ollama (se rodando localmente) ou o modo heurístico estático de fallback sem interromper o funcionamento.

---

## 7. Execução & Uso do Dashboard Streamlit

Para iniciar a aplicação web iterativa:

```bash
streamlit run app.py
```

Após o comando, acesse o endereço no seu navegador (geralmente `http://localhost:8501`).

### Guia das Abas da Interface

1. **Aba "🔍 Search & Scrape"**:
   - Defina as palavras-chave, localização, modalidade de trabalho e plataformas desejadas.
   - Carregue seu currículo mestre em formato PDF, TXT ou colar texto.
   - Clique em **"🚀 Trigger Scrape & Search"** para iniciar a captura de vagas e o cálculo imediato de aderência.

2. **Aba "📊 Matched Jobs"**:
   - Filtre as vagas por nota mínima de compatibilidade (*Match Score*), palavras-chave, veredito (`STRONG_MATCH`, `APPLY`) ou status da aplicação.
   - Visualize os requisitos correspondentes e faltantes para cada vaga.
   - Atualize o status da candidatura (*Saved*, *Applied*, *Interviewing*, *Offer*).

3. **Aba "📝 Application Kit"**:
   - Selecione qualquer vaga salva.
   - Clique em **"✨ Generate / Re-generate Application Kit"**.
   - Obtenha a Carta de Apresentação (*Cover Letter*) personalizada e os pontos de conquista para currículo, revisados pelo Agente Auditor com nota de compatibilidade ATS.

---

## 8. Suíte de Testes Automatizados

O projeto possui uma suíte completa de testes automatizados com `pytest` cobrindo o banco de dados, deduplicação, fallback de LLM, parser de JSON e pipeline de agentes:

```bash
# Executar todos os testes unitários e de integração
python3 -m pytest tests/ -v
```

### Resultados Esperados:
```
job_ai_app/tests/test_job_ai_app.py::test_compute_job_hash PASSED
job_ai_app/tests/test_job_ai_app.py::test_database_crud_and_deduplication PASSED
job_ai_app/tests/test_job_ai_app.py::test_extract_json_from_text PASSED
job_ai_app/tests/test_job_ai_app.py::test_unified_llm_client_fallback_chain PASSED
job_ai_app/tests/test_job_ai_app.py::test_heuristic_fallback_fit PASSED
job_ai_app/tests/test_job_ai_app.py::test_evaluate_job_fit_fallback PASSED
job_ai_app/tests/test_job_ai_app.py::test_drafter_and_reviewer_agents PASSED
job_ai_app/tests/test_job_ai_app.py::test_fetch_job_listings_mocked PASSED
```

---

## 9. Segurança & Boas Práticas

- **Proteção de Credenciais**: Nenhuma chave de API é versionada no repositório. O `.env` está explicitamente no `.gitignore`.
- **Privacidade de Dados do Usuário**: Seu currículo e dados de vagas são armazenados estritamente na sua instância SQLite local (`job_ai_app.db`).
- **Validação de Schemas Estritos**: Uso intensivo de `pydantic` para prevenir alucinações e respostas malformadas de LLMs.

---

## 10. Troubleshooting

### 1. `ModuleNotFoundError: No module named 'jobspy'`
- **Causa**: As dependências do projeto não foram instaladas no ambiente virtual correto.
- **Solução**: Execute `pip install -r requirements.txt`.

### 2. Bloqueios de Rede durante o Scraping (ex: Código 403 no ZipRecruiter ou Glassdoor)
- **Causa**: Proteções de Cloudflare/WAF dos portais de vagas contra acessos automatizados frequentes.
- **Solução**: O JobSpy já gerencia proxies e headers temporários. O `job_ai_app` trata erros de requisição graciosamente mantendo os resultados capturados dos demais sites (ex: LinkedIn e Indeed).

### 3. Falha de Conexão com o Ollama (`http://localhost:11434`)
- **Causa**: O serviço local do Ollama não está rodando.
- **Solução**: Inicie o servidor Ollama com `ollama serve` ou configure uma chave de API para o Groq/Gemini no menu lateral do Streamlit.
