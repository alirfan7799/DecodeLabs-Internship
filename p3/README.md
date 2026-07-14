# Career & Job Recommender — Semantic Edition

A content-based recommendation engine that matches a person's described
background to real job postings using **TF-IDF + Truncated SVD (Latent
Semantic Analysis)** — no hand-curated skill list, no fixed vocabulary.

Built as the Project 3 capstone (DecodeLabs Industrial Training Kit, AI
Engineer track: *AI Recommendation Logic*) — evolved through two full
review cycles based on real, verified engineering flaws (documented below).

## Why this isn't a toy project

Runs on **1,250 real Google job postings** across **23 job categories**
(Software Engineering, Sales, Marketing, Legal, UX Design, Finance, Program
Management, and more). No pre-labeled skills column — just raw,
unstructured `Responsibilities` / `Minimum Qualifications` / `Preferred
Qualifications` text, exactly like a production system actually receives.

## Architecture

All files sit flat in one folder — no subfolders.

```
job_skills.csv (raw, unstructured)
        │
        ▼
data_pipeline.py     — Ingestion: combines text fields, filters
                        near-empty postings. No manual skill extraction.
        │
        ▼
semantic_engine.py   — Process:
                        1. TfidfVectorizer learns its vocabulary directly
                           from the corpus (unigrams + bigrams, ~25k terms)
                        2. TruncatedSVD compresses that into a 150-200
                           dimension latent semantic space — terms that
                           co-occur in similar contexts end up with
                           similar vectors (this is what lets "container
                           orchestration" match "Kubernetes" postings
                           without exact word overlap)
                        3. User's free text is projected into the SAME
                           learned space and ranked via cosine similarity
                        — Output: dedup by (Title, Company), grouped
                          locations, Top-N filtering, cold-start fallback
        │
        ▼
app.py                — Streamlit UI: free-text background input,
                         ranked match cards with real evidence sentences,
                         score chart, shared-term breakdown, category
                             alignment chart
```

## Evolution history — the flaws that shaped this design

This project went through two real rounds of failure-finding, not just
one clean build. Both are documented here on purpose — a system that's
survived actual bug hunts is more defensible than one that's never been
stress-tested.

**Round 1 — original hand-curated taxonomy + raw cosine similarity:**
A query of `python, c++, java, git` returned a Marketing posting ranked
*above* an actual Software Engineering Manager role. Root cause: `git` was
a rare term across the corpus, so TF-IDF gave it a disproportionate weight,
and one rare shared word dominated the score regardless of how much of the
user's actual profile the posting covered. Fixed at the time with a
skill-coverage blending heuristic — but the deeper issue was the fixed
150-term taxonomy itself: it can't recognize any skill I didn't manually
type in, and can't capture that "deep learning" and "neural networks" are
related concepts.

**Round 2 — rebuilding on Latent Semantic Analysis:**
Replaced the taxonomy entirely with an auto-learned vocabulary. First pass
still failed: a query about "Go, Kubernetes, container orchestration, cloud
infrastructure" returned **Sales and Partnerships roles** ahead of
Engineering ones. Diagnosed two exact causes:
1. `stop_words="english"` silently deletes "go" (treated as the common
   English word, not the Go language) before it can ever be counted.
2. `max_features=6000` with bigrams enabled meant the vocabulary was capped
   low enough that rare-but-highly-specific terms (`kubernetes`, `docker`,
   `backend` — each appearing in only 3-6 of 1,236 postings) got crowded out
   by generic high-frequency terms, while generic overlap ("cloud",
   "services", "managing") dominated the similarity score instead.

Fixed by building a custom stopword list (standard English list minus terms
that double as real tech vocabulary) and raising `max_features` to 25,000 /
lowering `min_df` to 2, so rare specific terms survive vocabulary
construction instead of being crowded out. Re-verified against the exact
same failing query afterward — Software Engineering and Cloud
Infrastructure roles now rank first, with a directly checkable evidence
sentence quoting the actual posting text.

**Round 3 — the "AI engineer" transparency gap:**
A query of `"I am an AI engineer"` returned Cloud Sales roles at 40-45%
"match" with no visible reason why. Diagnosed the exact cause: this
dataset (a 2017-2018 snapshot) contains the literal abbreviation "AI"
**zero times** — postings always spell out "artificial intelligence"
instead. After stopword removal, the entire 5-word query collapsed down to
a single surviving term: "engineer" — generic enough to appear in Sales
Engineer, Solutions Engineer, and Software Engineer postings alike, so it
couldn't discriminate between them. The score wasn't mathematically wrong
given the information available — the real problem was that nothing told
the user their query had silently degraded to one generic word.

The first fix attempt used a small hardcoded abbreviation-expansion table
(`AI` -> `artificial intelligence`, etc.). That was deliberately reverted —
hardcoding substitutions hides the real limitation instead of surfacing it,
and doesn't scale to every word a system might not recognize. The honest
fix instead: a **word-by-word transparency breakdown** shown directly in
the UI ("How your search was actually interpreted"). Every word in the
query is classified as `matched` (a real term found in this dataset),
`not_in_dataset` (a substantive word that genuinely never appears anywhere
in these 1,236 postings — a data coverage limit, not a system bug), or
`filler` (a common word like "the"/"an" carrying no signal either way).
A visible match ratio (e.g. "1/2 words found — 50%") and a warning banner
fire whenever a result is being driven by too little real signal. This is
a genuine data limitation as much as a design choice: only 45 of 1,236
postings mention ML/AI/deep learning at all, and none are literally titled
"AI Engineer" — the system now tells the truth about what it actually
found in the data, rather than silently guessing or quietly patching over
individual missing words one at a time.

**Round 4 — removing the last hand-authored content:**
The "Try an example" picker originally had 23 example sentences — one per
category — that I wrote by hand as UI convenience text. This never touched
the recommendation engine (it only pre-filled a text box you could freely
overwrite), but it was still hand-authored content in a project whose
entire philosophy is "let the real data speak." Replaced with a function
(`build_category_examples` in `app.py`) that pulls an actual sentence
directly from a real posting in each category — specifically the posting
closest to that category's median word count, to avoid picking a freak
outlier — so every example a user sees is real, verifiable text from the
dataset itself, not something typed by hand.

**Round 5 — the trivial self-match:**
Immediately after Round 4, every example query was scoring 90%+ against
its own source posting. This is mathematically guaranteed, not a bug: a
query built from a document's own literal words will always score
near-perfect against that exact document under cosine similarity. It
proved nothing about real query performance, since a real person's typed
description is never word-for-word identical to a posting.

The first attempt (combining sentences from two different postings) didn't
fix it — it just produced two near-perfect self-matches instead of one,
since each half still contained its own source's literal words verbatim.
The real fix: `recommend()` now accepts an `exclude_indices` parameter, and
when an example's exact text is submitted, its source posting is excluded
from the ranking entirely — the query text stays 100% real, unedited
corpus text, but the results reflect genuine similarity to *other*
postings, not an echo of itself.

This surfaced a second, more interesting layer underneath: excluding one
row wasn't enough either, because Google reposts the identical boilerplate
text for the same role across multiple office locations (5 near-duplicate
"Software Engineer" postings were found sharing one phrase verbatim). Fixed
by excluding the entire `(Title, Company)` group — the same equivalence
class the dedup logic elsewhere in the code already uses — rather than a
single row index.

One residual case remains, and it's correct behavior rather than something
to fix: a *differently*-titled posting ("Software Engineer (English)")
still scores 94% against the example. Checked directly — the literal
shared TF-IDF terms are weak and generic (`improve`, `software`,
`project`, `develop`, all with tiny weights), so the score isn't coming
from copied text at all. It's coming from the 200-dimension latent space
correctly recognizing that two differently-worded but genuinely similar
"Software Engineer" postings occupy the same thematic region — which is
exactly what LSA is supposed to do. Two real postings for the same broad
role really are that similar in this dataset; suppressing that score
would be hiding a true finding, not fixing a flaw.

**Round 6 — abandoning category-as-query entirely:**
The next idea was to simplify further: let selecting a category from the
sidebar directly run a search using the category name itself as the query
(e.g. searching literally "Software Engineering"), removing the need to
pick example text at all. Tested systematically across all 23 categories
before shipping it — and it failed for 9 of them, including "Software
Engineering" itself (0% of its own top-8 results were actually tagged
Software Engineering) and "IT & Data Management" (also 0%). Root cause:
words like "engineering" and "technical" are too scattered across many
different categories in this specific corpus to reliably steer a search
back to their own label — a category name is a label, not a description,
and asking the similarity engine to treat it as one doesn't hold up.

The fix was architectural, not another parameter tweak: category browsing
and semantic search were split into two genuinely separate features,
matching how real job platforms handle this (browsing a category and
searching your own profile are different actions, not one disguised as
the other). `browse_category()` in `semantic_engine.py` is a plain filter
on the real `Category` column — no scoring, no query text, no possibility
of the failure modes above, because there's no similarity computation
involved at all. Selecting a category now shows genuine postings from
that category with 100% precision by definition. The free-text "write
your own" path still runs the full TF-IDF + SVD + cosine pipeline exactly
as before, untouched.

## Design decisions worth knowing

- **LSA over raw TF-IDF cosine**: raw keyword overlap can't recognize that
  two related-but-differently-worded skills mean the same thing. Truncated
  SVD compresses the TF-IDF space into latent dimensions that capture
  co-occurrence patterns across the whole corpus — the classical technique
  underlying modern embedding models.
- **Why not transformer embeddings (e.g. sentence-transformers)?**
  Genuinely the stronger long-term option, and documented as the natural
  next upgrade below — but pretrained transformer models are downloaded
  from Hugging Face at runtime, and that domain wasn't reachable from the
  sandbox this was built in. Rather than ship an integration I couldn't
  personally verify end-to-end, LSA was chosen because it's fully
  self-contained and testable with zero external dependencies. It will run
  fine on your machine with normal internet access if you want to attempt
  the upgrade (see below).
- **Evidence sentences over abstract scores**: a 150-dimension latent
  similarity score isn't human-readable on its own, so every match also
  surfaces the actual sentence from the posting that most overlaps with
  the user's text — concrete, checkable proof instead of a black-box number.
- **Deduplication by (Title, Company)**: the raw dataset posts the same
  role multiple times across different office locations. Results now group
  those into one card with all locations listed, instead of repeating
  visually identical cards.

## Honest limitations that remain

- **Content-based only** — no collaborative filtering, no learning from
  clicks/feedback over time. Every recommendation depends entirely on what
  the user types in that session.
- **Static dataset snapshot** — no live postings, no salary/seniority
  weighting.
- **LSA explained variance is ~60-65%** — meaningful signal, but not a
  complete picture of the corpus's semantic structure. A denser transformer
  embedding would likely capture more nuance.

## Optional upgrade path: transformer embeddings

If you want to push this further, `sentence-transformers` (e.g.
`all-MiniLM-L6-v2`) can replace the LSA layer with dense neural embeddings
that generalize even better across paraphrased or unusual phrasing. It
wasn't shipped here because it couldn't be verified in this environment,
but the integration point is exactly `semantic_engine.py`'s `_embed_user_text`
and the one-time corpus embedding step — swap TF-IDF+SVD vectors for
`model.encode(...)` output and the rest of the pipeline (ranking, dedup,
cold-start) is unaffected.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Dataset

`job_skills.csv` — Google job postings dataset (Company, Title,
Category, Location, Responsibilities, Minimum Qualifications, Preferred
Qualifications). 1,250 rows; 1,236 retained after filtering near-empty text.
