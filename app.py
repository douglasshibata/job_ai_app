import os
import io
import json
import pandas as pd
import streamlit as st
from pypdf import PdfReader

try:
    from job_ai_app.config import (
        DEFAULT_SEARCH_TERM, DEFAULT_LOCATION, DEFAULT_HOURS_OLD,
        DEFAULT_RESULTS_WANTED, DEFAULT_SITES, GROQ_API_KEY, GEMINI_API_KEY,
        OLLAMA_HOST, OLLAMA_MODEL, PRIMARY_LLM_PROVIDER
    )
    from job_ai_app.database import (
        init_db, get_jobs, get_job_by_id, update_job_fit,
        save_application_kit, get_application_kit_by_job_id, update_job_status
    )
    from job_ai_app.scraper import fetch_job_listings
    from job_ai_app.llm_client import UnifiedLLMClient, GroqProvider, GeminiProvider, OllamaProvider
    from job_ai_app.matcher import evaluate_job_fit, generate_application_kit, review_application_kit
except ImportError:
    from config import (
        DEFAULT_SEARCH_TERM, DEFAULT_LOCATION, DEFAULT_HOURS_OLD,
        DEFAULT_RESULTS_WANTED, DEFAULT_SITES, GROQ_API_KEY, GEMINI_API_KEY,
        OLLAMA_HOST, OLLAMA_MODEL, PRIMARY_LLM_PROVIDER
    )
    from database import (
        init_db, get_jobs, get_job_by_id, update_job_fit,
        save_application_kit, get_application_kit_by_job_id, update_job_status
    )
    from scraper import fetch_job_listings
    from llm_client import UnifiedLLMClient, GroqProvider, GeminiProvider, OllamaProvider
    from matcher import evaluate_job_fit, generate_application_kit, review_application_kit

# Initialize Database Schema
init_db()

# Page Setup
st.set_page_config(
    page_title="AI Job Search & Application Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Helper: Read Master Resume from Uploaded File or Text Area
def extract_resume_text(uploaded_file, text_input: str) -> str:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".pdf"):
                pdf_reader = PdfReader(uploaded_file)
                text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
                if text.strip():
                    return text.strip()
            elif uploaded_file.name.endswith(".txt") or uploaded_file.name.endswith(".md"):
                return uploaded_file.getvalue().decode("utf-8").strip()
        except Exception as e:
            st.error(f"Error reading uploaded resume file: {e}")
    return text_input.strip()


# Sidebar Configuration
st.sidebar.title("⚙️ System Control")
st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Free LLM Configuration")
selected_provider = st.sidebar.selectbox(
    "Primary LLM Provider",
    ["groq", "gemini", "ollama"],
    index=["groq", "gemini", "ollama"].index(PRIMARY_LLM_PROVIDER) if PRIMARY_LLM_PROVIDER in ["groq", "gemini", "ollama"] else 0
)

groq_key_input = st.sidebar.text_input("Groq API Key", value=GROQ_API_KEY, type="password")
gemini_key_input = st.sidebar.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
ollama_host_input = st.sidebar.text_input("Ollama Host URL", value=OLLAMA_HOST)

# Initialize Unified Client with custom user settings
unified_llm = UnifiedLLMClient(primary_provider=selected_provider)
unified_llm.providers["groq"] = GroqProvider(api_key=groq_key_input)
unified_llm.providers["gemini"] = GeminiProvider(api_key=gemini_key_input)
unified_llm.providers["ollama"] = OllamaProvider(host=ollama_host_input)

# Check active providers
chain = unified_llm.get_provider_chain()
active_providers = [name for name, prov in chain if prov.is_available()]
if active_providers:
    st.sidebar.success(f"Active Provider(s): {', '.join(active_providers)}")
else:
    st.sidebar.info("Using heuristic/template fallback mode (No API keys configured).")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Master Resume / CV")
uploaded_resume_file = st.sidebar.file_uploader("Upload Resume (PDF / TXT / MD)", type=["pdf", "txt", "md"])
resume_text_area = st.sidebar.text_area("Or Paste Master Resume Text", height=200, placeholder="Paste your CV text here...")
master_resume = extract_resume_text(uploaded_resume_file, resume_text_area)

if master_resume:
    st.sidebar.success(f"Resume Loaded ({len(master_resume)} characters)")
else:
    st.sidebar.warning("Please upload or paste your Master Resume for AI matching.")


# Main Application Interface
st.markdown("<div class='main-header'>🎯 AI Job Search & Application Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Automated Multi-Board Job Scraping, AI Fit Analysis & Drafter-Reviewer Application Kit</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Search & Scrape", "📊 Matched Jobs", "📝 Application Kit"])


# ==========================================
# TAB 1: SEARCH & SCRAPE
# ==========================================
with tab1:
    st.subheader("Multi-Board Job Scraping Engine")
    st.write("Query LinkedIn, Indeed, Glassdoor, and ZipRecruiter concurrently using JobSpy.")

    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Job Title / Keywords", value=DEFAULT_SEARCH_TERM)
        location = st.text_input("Location", value=DEFAULT_LOCATION)
        work_type = st.selectbox("Work Type", ["All", "Remote", "Hybrid", "Onsite"])

    with col2:
        hours_old_map = {"Past 24 Hours": 24, "Past 3 Days": 72, "Past Week": 168}
        hours_selected = st.selectbox("Date Posted Filter", list(hours_old_map.keys()), index=1)
        hours_old = hours_old_map[hours_selected]
        results_wanted = st.slider("Max Results Per Site", min_value=5, max_value=50, value=DEFAULT_RESULTS_WANTED, step=5)
        selected_sites = st.multiselect("Job Boards to Scrape", ["linkedin", "indeed", "glassdoor", "zip_recruiter"], default=DEFAULT_SITES)

    auto_evaluate = st.checkbox("Automatically run AI Fit Evaluation on newly scraped jobs against Master Resume", value=True)

    if st.button("🚀 Trigger Scrape & Search", type="primary", use_container_width=True):
        if not selected_sites:
            st.error("Please select at least one job board to scrape.")
        else:
            with st.spinner("Scraping job boards concurrently via JobSpy... This may take a few seconds."):
                res = fetch_job_listings(
                    search_term=search_term,
                    location=location,
                    work_type=work_type if work_type != "All" else None,
                    hours_old=hours_old,
                    results_wanted=results_wanted,
                    sites=selected_sites
                )

            st.success(f"Scrape completed! Total scraped: {res['total_scraped']} | New inserted: {res['inserted']} | Skipped duplicates: {res['skipped']}")

            if auto_evaluate and master_resume and res['inserted'] > 0:
                st.info("Running AI Fit Evaluation on new job listings...")
                jobs_to_eval = get_jobs(verdict_filter="UNEVALUATED")
                eval_progress = st.progress(0)
                for idx, job in enumerate(jobs_to_eval):
                    fit_report = evaluate_job_fit(
                        job_description=job.description or f"{job.title} at {job.company}",
                        master_resume=master_resume,
                        llm_client=unified_llm
                    )
                    update_job_fit(job.id, fit_report.model_dump())
                    eval_progress.progress((idx + 1) / len(jobs_to_eval))
                st.success("AI Fit Evaluation complete! Head to the 'Matched Jobs' tab to view results.")

    st.markdown("---")
    st.subheader("Stored Job Listings Overview")
    stored_jobs = get_jobs()
    if stored_jobs:
        st.write(f"Total jobs stored in SQLite database: **{len(stored_jobs)}**")
        df_summary = pd.DataFrame([
            {
                "ID": j.id,
                "Title": j.title,
                "Company": j.company,
                "Location": j.location,
                "Site": j.site,
                "Match Score": f"{j.match_score}%",
                "Verdict": j.verdict,
                "Status": j.status
            } for j in stored_jobs[:10]
        ])
        st.dataframe(df_summary, use_container_width=True)
    else:
        st.info("No job listings in the database yet. Trigger a scrape above!")


# ==========================================
# TAB 2: MATCHED JOBS
# ==========================================
with tab2:
    st.subheader("Job Matching & Fit Dashboard")

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        min_score = st.slider("Min Match Score Filter", min_value=0, max_value=100, value=0, step=5)
    with fcol2:
        search_kw = st.text_input("Filter Keyword", value="", placeholder="e.g. Python, Senior...")
    with fcol3:
        verdict_flt = st.selectbox("Verdict Filter", ["ALL", "STRONG_MATCH", "APPLY", "PASS", "UNEVALUATED"])
    with fcol4:
        status_flt = st.selectbox("Status Filter", ["ALL", "Saved", "Applied", "Interviewing", "Rejected", "Offer"])

    filtered_jobs = get_jobs(
        min_match_score=min_score,
        search_query=search_kw,
        verdict_filter=verdict_flt,
        status_filter=status_flt
    )

    # Summary Metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total Listings", len(filtered_jobs))
    high_matches = [j for j in filtered_jobs if j.match_score >= 75]
    mcol2.metric("High Matches (>75%)", len(high_matches))
    avg_score = int(sum(j.match_score for j in filtered_jobs) / max(len(filtered_jobs), 1))
    mcol3.metric("Avg Match Score", f"{avg_score}%")
    applied_count = len([j for j in filtered_jobs if j.status == "Applied"])
    mcol4.metric("Applied Count", applied_count)

    st.markdown("---")

    if not filtered_jobs:
        st.warning("No jobs matching the selected filter criteria found.")
    else:
        for job in filtered_jobs:
            with st.expander(f"**{job.title}** at **{job.company}** — Score: {job.match_score}% | {job.verdict} | [{job.site.upper()}]"):
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    st.write(f"**Location:** {job.location} | **Source:** {job.site} | **Type:** {job.job_type or 'N/A'}")
                    if job.min_amount or job.max_amount:
                        st.write(f"**Salary:** {job.currency} {job.min_amount or 0:,.0f} - {job.max_amount or 0:,.0f}")

                    st.markdown("##### Fit Evaluation Summary")
                    st.write(job.fit_summary or "No summary generated yet.")

                    # Skills tags
                    matching_skills = json.loads(job.matching_skills or "[]")
                    missing_skills = json.loads(job.missing_skills or "[]")

                    if matching_skills:
                        st.markdown("**Matching Skills:** " + ", ".join([f"`{s}`" for s in matching_skills]))
                    if missing_skills:
                        st.markdown("**Missing Requirements:** " + ", ".join([f"`{s}`" for s in missing_skills]))

                    st.markdown("##### Job Description Preview")
                    st.text_area("Full Description", value=job.description or "No description available.", height=150, key=f"desc_{job.id}")

                with col_right:
                    st.markdown("##### Actions & Lifecycle")

                    # Status updater
                    new_st = st.selectbox(
                        "Application Status",
                        ["Saved", "Applied", "Interviewing", "Rejected", "Offer"],
                        index=["Saved", "Applied", "Interviewing", "Rejected", "Offer"].index(job.status) if job.status in ["Saved", "Applied", "Interviewing", "Rejected", "Offer"] else 0,
                        key=f"status_sel_{job.id}"
                    )
                    if new_st != job.status:
                        update_job_status(job.id, new_st)
                        st.success(f"Status updated to '{new_st}'")

                    if job.job_url:
                        st.markdown(f"[🔗 Open Original Job Listing]({job.job_url})")

                    if st.button("⚡ Evaluate Fit with AI", key=f"eval_btn_{job.id}"):
                        if not master_resume:
                            st.error("Please load your Master Resume in the sidebar first.")
                        else:
                            with st.spinner("Analyzing resume fit..."):
                                fit = evaluate_job_fit(job.description or job.title, master_resume, unified_llm)
                                update_job_fit(job.id, fit.model_dump())
                            st.success("Fit re-evaluated!")
                            st.rerun()


# ==========================================
# TAB 3: APPLICATION KIT
# ==========================================
with tab3:
    st.subheader("Drafter-Reviewer Application Kit Generator")
    st.write("Generate tailored cover letters and customized resume achievement bullets audited against your base resume.")

    all_jobs = get_jobs()
    if not all_jobs:
        st.info("No jobs available in database to generate application kits.")
    else:
        # Create job selection options
        job_options = {f"[{j.match_score}% Match] {j.title} - {j.company} (ID: {j.id})": j.id for j in all_jobs}
        selected_option = st.selectbox("Select Target Job", list(job_options.keys()))
        selected_job_id = job_options[selected_option]
        target_job = get_job_by_id(selected_job_id)

        if target_job:
            st.markdown(f"### Target Role: **{target_job.title}** at **{target_job.company}**")

            existing_kit = get_application_kit_by_job_id(target_job.id)

            if st.button("✨ Generate / Re-generate Application Kit", type="primary"):
                if not master_resume:
                    st.error("Please upload or paste your Master Resume in the sidebar before generating application materials.")
                else:
                    with st.spinner("Drafter Agent creating tailored cover letter and customized bullets..."):
                        drafter_output = generate_application_kit(
                            job_title=target_job.title,
                            company=target_job.company,
                            job_description=target_job.description or f"{target_job.title} at {target_job.company}",
                            master_resume=master_resume,
                            llm_client=unified_llm
                        )

                    with st.spinner("Reviewer Agent auditing generated materials against Master Resume..."):
                        reviewer_output = review_application_kit(
                            master_resume=master_resume,
                            job_description=target_job.description or f"{target_job.title} at {target_job.company}",
                            cover_letter=drafter_output.tailored_cover_letter,
                            customized_bullets=drafter_output.customized_bullets,
                            llm_client=unified_llm
                        )

                    # Save to database
                    save_application_kit(
                        job_id=target_job.id,
                        tailored_cover_letter=drafter_output.tailored_cover_letter,
                        customized_bullets=drafter_output.customized_bullets,
                        audit_status=reviewer_output.audit_status,
                        audit_notes=reviewer_output.audit_notes
                    )

                    st.success("Application Kit generated and audited successfully!")
                    st.rerun()

            kit = get_application_kit_by_job_id(target_job.id)
            if kit:
                st.markdown("---")

                # Reviewer Audit Header
                ocol1, ocol2, ocol3 = st.columns(3)
                with ocol1:
                    st.metric("Audit Status", kit.audit_status)
                with ocol2:
                    st.metric("Reviewer Notes", kit.audit_notes or "Passed compliance checks.")
                with ocol3:
                    if st.button("Mark as Applied 🚀"):
                        update_job_status(target_job.id, "Applied")
                        st.success("Job status updated to 'Applied'!")

                st.markdown("---")
                st.subheader("✉️ Tailored Cover Letter")
                editable_letter = st.text_area("Edit Cover Letter", value=kit.tailored_cover_letter, height=300)

                st.download_button(
                    label="📥 Download Cover Letter (.txt)",
                    data=editable_letter,
                    file_name=f"Cover_Letter_{target_job.company.replace(' ', '_')}_{target_job.title.replace(' ', '_')}.txt",
                    mime="text/plain"
                )

                st.markdown("---")
                st.subheader("🎯 Customized Resume Achievement Bullets")
                try:
                    bullets_list = json.loads(kit.customized_bullets)
                except Exception:
                    bullets_list = [kit.customized_bullets]

                for bullet in bullets_list:
                    st.markdown(f"- {bullet}")

                st.download_button(
                    label="📥 Download Bullets (.txt)",
                    data="\n".join([f"- {b}" for b in bullets_list]),
                    file_name=f"Resume_Bullets_{target_job.company.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
