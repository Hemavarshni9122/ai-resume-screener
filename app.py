# ============================================================
# app.py — The Web App (Streamlit Frontend)
# 
# HOW TO RUN:
#   streamlit run app.py
#
# Streamlit works like this:
#   - Every time the user interacts (clicks, types), 
#     the entire script re-runs top to bottom.
#   - st.session_state stores data between re-runs.
#   - Each st.xxx() call creates a UI element.
# ============================================================

import streamlit as st
import pandas as pd
from ranker import rank_resumes, extract_keywords

# -------------------------------------------------------
# PAGE CONFIGURATION
# Must be the FIRST streamlit command in the file.
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------------------------------
# CUSTOM CSS — Makes the app look professional
# -------------------------------------------------------
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .title { font-size: 2.5rem; font-weight: 800; color: #1a1a2e; }
    .subtitle { color: #555; font-size: 1rem; margin-bottom: 2rem; }
    .rank-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #4f8ef7;
    }
    .rank-1 { border-left-color: #FFD700; }
    .rank-2 { border-left-color: #C0C0C0; }
    .rank-3 { border-left-color: #CD7F32; }
    .score-badge {
        background: #4f8ef7;
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .keyword-tag {
        background: #e8f4f8;
        color: #2c7be5;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown('<div class="title">🤖 AI-Powered Resume Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste a job description and add candidate resumes — the system ranks them by relevance using NLP & TF-IDF Cosine Similarity.</div>', unsafe_allow_html=True)
st.markdown("---")


# -------------------------------------------------------
# LAYOUT — Two columns side by side
# col1 = inputs (job description + resumes)
# col2 = results (ranked output)
# -------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")


# ===================== LEFT COLUMN =====================
with col1:
    st.subheader("📋 Job Description")
    
    # Text area for job description input
    # The 'value' shown as placeholder helps users understand what to paste
    job_description = st.text_area(
        label="Paste the job description here",
        height=200,
        placeholder="e.g. We are looking for a Python developer with experience in machine learning, data analysis, scikit-learn, pandas, and SQL. The candidate should have knowledge of NLP and computer vision...",
        key="jd_input"
    )

    st.subheader("👥 Add Candidates")
    st.caption("Enter each candidate's name and their resume text. Add up to 10 candidates.")

    # -------------------------------------------------------
    # DYNAMIC CANDIDATE ADDITION
    # We store candidate list in session_state so it 
    # persists when Streamlit re-runs the script.
    # -------------------------------------------------------
    if 'candidates' not in st.session_state:
        # Initialize with 2 empty candidates on first load
        st.session_state.candidates = [
            {"name": "", "resume": ""},
            {"name": "", "resume": ""}
        ]

    # Render input fields for each candidate
    for i, candidate in enumerate(st.session_state.candidates):
        with st.expander(f"Candidate {i+1}: {candidate['name'] or 'Unnamed'}", expanded=(i < 2)):
            
            # Name input
            name = st.text_input(
                f"Full Name",
                value=candidate["name"],
                key=f"name_{i}",
                placeholder="e.g. Hemavarshni S"
            )
            
            # Resume text input
            resume_text = st.text_area(
                f"Resume Text",
                value=candidate["resume"],
                key=f"resume_{i}",
                height=150,
                placeholder="Paste the full resume text here (copy from PDF/Word)..."
            )
            
            # Update session state
            st.session_state.candidates[i]["name"] = name
            st.session_state.candidates[i]["resume"] = resume_text

    # Button to add more candidates
    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("➕ Add Candidate", use_container_width=True):
            if len(st.session_state.candidates) < 10:
                st.session_state.candidates.append({"name": "", "resume": ""})
                st.rerun()

    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.candidates = [{"name": "", "resume": ""}, {"name": "", "resume": ""}]
            st.rerun()

    # -------------------------------------------------------
    # MAIN ACTION BUTTON
    # When clicked, triggers the ranking logic
    # -------------------------------------------------------
    st.markdown("")
    run_button = st.button("🚀 Screen & Rank Resumes", type="primary", use_container_width=True)


# ===================== RIGHT COLUMN =====================
with col2:
    st.subheader("📊 Ranking Results")

    if run_button:
        # --- Validation ---
        if not job_description.strip():
            st.error("⚠️ Please paste a job description first.")
        else:
            # Build resumes dict — only include candidates with both name and text
            resumes = {
                c["name"].strip(): c["resume"].strip()
                for c in st.session_state.candidates
                if c["name"].strip() and c["resume"].strip()
            }

            if len(resumes) < 2:
                st.warning("⚠️ Please add at least 2 candidates with names and resume text.")
            else:
                # --- Run the ML ranking ---
                with st.spinner("🔍 Analysing resumes with TF-IDF..."):
                    results_df = rank_resumes(job_description, resumes)

                st.success(f"✅ Ranked {len(resumes)} candidates successfully!")
                st.markdown("")

                # --- Display results as cards ---
                rank_colors = {1: "rank-1", 2: "rank-2", 3: "rank-3"}

                for _, row in results_df.iterrows():
                    rank = int(row['Rank'])
                    rank_class = rank_colors.get(rank, "rank-card")
                    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

                    # Score bar width (percentage)
                    score_pct = float(row['Score']) * 100

                    st.markdown(f"""
                        <div class="rank-card {rank_class}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size:1.3rem;">{medal}</span>
                                    <strong style="font-size:1.1rem; margin-left:8px;">{row['Candidate']}</strong>
                                </div>
                                <span class="score-badge">{row['Match %']}</span>
                            </div>
                            <div style="margin-top:8px; background:#eee; border-radius:10px; height:8px;">
                                <div style="width:{score_pct:.1f}%; background:#4f8ef7; height:8px; border-radius:10px;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Show keyword matches for top 3
                    if rank <= 3:
                        resume_text = resumes[row['Candidate']]
                        keywords = extract_keywords(job_description, resume_text, top_n=8)
                        if keywords:
                            kw_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
                            st.markdown(f"<div style='margin: -8px 0 12px 20px;'>🔑 Key matches: {kw_html}</div>", unsafe_allow_html=True)

                # --- Download Results as CSV ---
                st.markdown("---")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="resume_ranking_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    else:
        # Placeholder when no results yet
        st.info("👈 Fill in the job description and candidate resumes on the left, then click **Screen & Rank Resumes**.")
        
        st.markdown("### 💡 How it works")
        st.markdown("""
        1. **Paste** a job description (skills, requirements, responsibilities)
        2. **Add** each candidate's name and paste their resume text
        3. **Click** Screen & Rank — the AI compares them using:
           - **TF-IDF**: converts text to numbers
           - **Cosine Similarity**: measures how closely each resume matches the JD
        4. **Results** are ranked from best match to lowest with a match score
        """)

        st.markdown("### 📐 Scoring Guide")
        score_data = {
            "Match %": ["80–100%", "60–79%", "40–59%", "Below 40%"],
            "Meaning": ["Excellent match", "Good match", "Partial match", "Low match"],
            "Action": ["Shortlist", "Consider", "Review carefully", "Likely not fit"]
        }
        st.table(pd.DataFrame(score_data))
