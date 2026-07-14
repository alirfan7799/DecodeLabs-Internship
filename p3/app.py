# Career & Job Recommender - semantic search (TF-IDF + SVD) over real Google job postings.
# Category browsing and free-text search are kept as two separate modes -
# using a category name as a search query was tried and dropped (see README).

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from data_pipeline import load_and_process
from semantic_engine import SemanticRecommender

DATA_PATH = os.path.join(os.path.dirname(__file__), "job_skills.csv")

st.set_page_config(
    page_title="Career & Job Recommender",
    page_icon="🧭",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_processed_data():
    return load_and_process(DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_engine(_df):
    return SemanticRecommender(_df)


st.markdown("""
<style>
    .match-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 5px solid #4f46e5;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .match-card.browse { border-left-color: #10b981; }
    .match-card h4 { margin: 0 0 4px 0; color: #1f2937; }
    .match-meta { color: #6b7280; font-size: 0.88rem; margin-bottom: 6px; }
    .match-score { font-weight: 700; color: #4f46e5; font-size: 1.05rem; }
    .evidence-box {
        background: #f8f9fb; border-left: 3px solid #10b981;
        padding: 8px 12px; margin-top: 8px; font-size: 0.85rem;
        color: #374151; border-radius: 4px; font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧭 Career & Job Recommender")
st.caption(
    "Semantic content-based filtering over **1,250+ real Google job postings** · "
    "TF-IDF + Latent Semantic Analysis · no fixed skill list — describe yourself freely"
)

with st.spinner("Loading job postings and building the semantic vector space..."):
    jobs_df = get_processed_data()
    engine = get_engine(jobs_df)

ALL_CATEGORIES = sorted(jobs_df["Category"].unique().tolist())

# sidebar
st.sidebar.header("Find a Role")

mode_choice = st.sidebar.selectbox(
    "Browse a category directly, or describe your own background",
    ["— write your own —"] + ALL_CATEGORIES,
)

BROWSE_MODE = mode_choice != "— write your own —"

top_n = st.sidebar.slider("Number of results", min_value=3, max_value=20, value=8)

if not BROWSE_MODE:
    st.sidebar.caption(
        "Write freely — skills, tools, experience, even a couple of sentences "
        "from your resume. There's no fixed list to pick from."
    )
    user_text = st.sidebar.text_area(
        "Your background",
        height=160,
        placeholder="e.g. I have 3 years of experience building REST APIs in Java, working with SQL databases, and deploying on AWS...",
    )
    run = st.sidebar.button("🔍 Find My Matches", type="primary", use_container_width=True)
else:
    st.sidebar.info(
        f"Showing real postings from **{mode_choice}** directly — this is a plain "
        "category filter, not a similarity search, so there's no match score involved."
    )
    run = True

st.sidebar.divider()
st.sidebar.metric("Postings indexed", len(jobs_df))
st.sidebar.metric("Job categories", jobs_df["Category"].nunique())
st.sidebar.metric("Vocabulary size (auto-learned)", len(engine.feature_names))

# browse mode - plain filter, no scoring
if BROWSE_MODE:
    st.subheader(f"Browsing: {mode_choice}")
    st.caption(
        f"These are real postings tagged **{mode_choice}** in the dataset — a direct "
        "filter, not a computed match. No score is shown because none is being computed."
    )
    browse_results = engine.browse_category(mode_choice, top_n=top_n)

    for _, row in browse_results.iterrows():
        loc_display = ", ".join(row["locations"][:3]) + (" + more" if len(row["locations"]) > 3 else "")
        preview = row["full_text"].strip().split(".")[0][:180]
        st.markdown(f"""
        <div class="match-card browse">
            <h4>{row['Title']}</h4>
            <div class="match-meta">{row['Company']} · {loc_display}</div>
            <div class="evidence-box">"{preview}..."</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# write your own mode - full semantic search pipeline
if not run:
    st.info(
        "👈 Describe your background in the sidebar in your own words, then click "
        "**Find My Matches** — or pick a category above to browse directly. The engine "
        "will semantically rank real Google job postings against your text — no keyword "
        "list required."
    )
    st.subheader("What's trending in the dataset right now")
    trending = engine.trending_categories(top_n=10)
    fig = px.bar(
        trending, x="posting_count", y="Category", orientation="h",
        color="posting_count", color_continuous_scale="Blues",
        labels={"posting_count": "Open Postings", "Category": ""},
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

if len(user_text.strip().split()) < 4:
    st.warning("Please write at least a short sentence describing your background — "
               "a few words gives the engine enough signal to work with.")
    st.stop()

result = engine.recommend(user_text, top_n=top_n)
results_df = result["results"]
diagnostics = engine.query_diagnostics(user_text)

with st.expander("🔍 How your search was actually interpreted", expanded=diagnostics["is_thin_query"]):
    st.write(f"**Your input:** \"{user_text}\"")

    if diagnostics["word_status"]:
        badge_map = {"matched": "✅", "not_in_dataset": "⚠️", "filler": "⏭️"}
        st.write("**Word-by-word breakdown:**")
        badges = "  ".join(
            f"{badge_map[w['status']]} `{w['word']}`" for w in diagnostics["word_status"]
        )
        st.markdown(badges)

        st.caption(
            "✅ matched a real term in this dataset  ·  "
            "⚠️ not found anywhere in this dataset (not this system's fault — "
            "it just isn't in these 1,236 postings)  ·  "
            "⏭️ common filler word, ignored either way"
        )

        st.metric(
            "Match ratio (substantive words actually found in the dataset)",
            f"{diagnostics['matched_count']} / {diagnostics['substantive_count']}"
            + (f"  ({diagnostics['match_ratio']:.0%})" if diagnostics["substantive_count"] else ""),
        )
    else:
        st.write("No words detected in your input.")

    if diagnostics["is_thin_query"]:
        st.warning(
            "⚠️ Very little of your input actually exists in this dataset — the "
            "ranking below is being driven by only one or two words, which can "
            "produce results that look off-topic. That's a **data coverage limit**, "
            "not a scoring bug: this dataset simply may not contain the specific "
            "words you used. Try rephrasing with different, more common terms, or "
            "adding more detail."
        )

if result["status"] == "cold_start":
    st.error(
        "**Cold Start detected** — nothing in your text matched the engine's learned "
        "vocabulary closely enough for personalized scoring. Showing trending postings "
        "instead (the standard fallback strategy for new/unmatched users)."
    )
    for _, row in results_df.head(top_n).iterrows():
        loc_display = ", ".join(row["locations"][:3]) + (" + more" if len(row["locations"]) > 3 else "")
        st.markdown(f"""
        <div class="match-card">
            <h4>{row['Title']}</h4>
            <div class="match-meta">{row['Company']} · {row['Category']} · {loc_display}</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ---- Ranked results ---- #
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.subheader(f"Top {len(results_df)} Matches")
    st.caption(
        "Scored by semantic similarity in a learned latent space, not exact keyword "
        "overlap — related concepts can match even without identical wording."
    )
    for idx, row in results_df.iterrows():
        pct = row["match_score"] * 100
        loc_display = ", ".join(row["locations"][:3]) + (" + more" if len(row["locations"]) > 3 else "")
        evidence = engine.evidence_sentence(idx, user_text)
        st.markdown(f"""
        <div class="match-card">
            <h4>{row['Title']}</h4>
            <div class="match-meta">{row['Company']} · {row['Category']} · {loc_display}</div>
            <div class="match-score">{pct:.1f}% semantic match</div>
            <div class="evidence-box">"{evidence}"</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.subheader("Match Score Comparison")
    chart_df = results_df.copy()
    chart_df["label"] = chart_df["Title"].str.slice(0, 40) + chart_df["Title"].apply(
        lambda t: "..." if len(t) > 40 else ""
    )
    chart_df["pct"] = chart_df["match_score"] * 100
    fig_scores = px.bar(
        chart_df.sort_values("pct"), x="pct", y="label", orientation="h",
        color="pct", color_continuous_scale="Purples",
        labels={"pct": "Match %", "label": ""},
    )
    fig_scores.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_scores, use_container_width=True)

    st.subheader("Shared Terms With Your Top Match")
    top_job_idx = results_df.index[0]
    breakdown = engine.explain_match(top_job_idx, result["user_tfidf"])
    if not breakdown.empty:
        fig_explain = px.bar(
            breakdown, x="weight", y="term", orientation="h",
            color="weight", color_continuous_scale="Teal",
            labels={"weight": "Shared Weight", "term": ""},
        )
        fig_explain.update_layout(showlegend=False, coloraxis_showscale=False, height=280)
        st.plotly_chart(fig_explain, use_container_width=True)
    else:
        st.caption(
            "No exact overlapping terms — this match was driven purely by semantic "
            "similarity in the latent space (related concepts, not identical wording)."
        )

st.divider()
st.subheader("Career Alignment Across Categories")
st.caption("Which job categories your background aligns with most, across all matches above the relevance threshold.")
cat_alignment = (
    results_df.groupby("Category")["match_score"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
fig_cat = px.pie(cat_alignment, names="Category", values="match_score", hole=0.45)
fig_cat.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig_cat, use_container_width=True)
