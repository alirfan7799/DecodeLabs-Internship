import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# sklearn's stopword list deletes "go" and "r" as common English words,
# but here they mean the Go and R programming languages
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - {"go", "r"})


class SemanticRecommender:
    def __init__(self, jobs_df: pd.DataFrame, n_components: int = 200, random_state: int = 42):
        self.jobs_df = jobs_df.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            stop_words=CUSTOM_STOP_WORDS,
            ngram_range=(1, 2),
            min_df=2,             # keep rare but specific terms (e.g. "kubernetes")
            max_df=0.6,           # drop terms too generic to be useful
            max_features=25000,   # kept high so bigrams don't crowd out rare terms
            sublinear_tf=True,
        )
        tfidf_matrix = self.vectorizer.fit_transform(self.jobs_df["full_text"])

        n_components = min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.item_latent = self.svd.fit_transform(tfidf_matrix)

        self.item_tfidf = tfidf_matrix  # kept only for explain_match(), not scoring
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.jobs_df["popularity_score"] = self.jobs_df["word_count"]

    def _embed_user_text(self, text: str):
        tfidf_vec = self.vectorizer.transform([text])
        latent_vec = self.svd.transform(tfidf_vec)
        return tfidf_vec, latent_vec

    def query_diagnostics(self, user_text: str) -> dict:
        """Classifies every typed word as matched / not_in_dataset / filler, so a search is never a black box."""
        raw_words = re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", user_text.lower())
        vocab_set = set(self.feature_names)

        word_status = []
        substantive_count = 0
        matched_count = 0

        for word in raw_words:
            if word in CUSTOM_STOP_WORDS:
                status = "filler"
            else:
                substantive_count += 1
                if word in vocab_set:
                    status = "matched"
                    matched_count += 1
                else:
                    status = "not_in_dataset"
            word_status.append({"word": word, "status": status})

        match_ratio = (matched_count / substantive_count) if substantive_count else 0.0

        return {
            "raw_word_count": len(raw_words),
            "word_status": word_status,
            "matched_count": matched_count,
            "substantive_count": substantive_count,
            "match_ratio": match_ratio,
            "is_thin_query": matched_count <= 1 or match_ratio < 0.4,
        }

    def recommend(self, user_text: str, top_n: int = 10, exclude_indices: list = None) -> dict:
        user_text = (user_text or "").strip()
        user_tfidf, user_latent = self._embed_user_text(user_text)

        if user_tfidf.nnz == 0 or not user_text:
            ranked = self.jobs_df.sort_values("popularity_score", ascending=False)
            fallback = ranked.drop_duplicates(subset=["Title", "Company"]).head(top_n).copy()
            fallback["match_score"] = np.nan
            fallback["locations"] = fallback["Location"].apply(lambda x: [x])
            return {"status": "cold_start", "results": fallback, "user_tfidf": user_tfidf}

        scores = cosine_similarity(user_latent, self.item_latent).flatten()

        all_results = self.jobs_df.copy()
        all_results["match_score"] = scores
        if exclude_indices:
            all_results = all_results.drop(index=[i for i in exclude_indices if i in all_results.index])
        all_results = all_results[all_results["match_score"] > 0.05]
        all_results = all_results.sort_values("match_score", ascending=False)

        # same posting reposted across locations -> group into one card
        deduped_rows = []
        seen = set()
        for _, row in all_results.iterrows():
            key = (row["Title"], row["Company"])
            if key in seen:
                continue
            seen.add(key)
            locations = all_results[
                (all_results["Title"] == row["Title"]) & (all_results["Company"] == row["Company"])
            ]["Location"].unique().tolist()
            row = row.copy()
            row["locations"] = locations
            deduped_rows.append(row)
            if len(deduped_rows) >= top_n:
                break

        results = pd.DataFrame(deduped_rows) if deduped_rows else all_results.head(0)
        return {"status": "ok", "results": results, "user_tfidf": user_tfidf}

    def explain_match(self, job_index: int, user_tfidf, top_k: int = 6) -> pd.DataFrame:
        """Shows real overlapping words behind a match, since the latent-space score itself isn't human-readable."""
        item_vec = self.item_tfidf[job_index].toarray().flatten()
        user_vec = user_tfidf.toarray().flatten()

        shared_weight = item_vec * user_vec
        nonzero = np.where(shared_weight > 0)[0]

        if len(nonzero) == 0:
            return pd.DataFrame(columns=["term", "weight"])

        return pd.DataFrame({
            "term": self.feature_names[nonzero],
            "weight": shared_weight[nonzero],
        }).sort_values("weight", ascending=False).head(top_k)

    def browse_category(self, category: str, top_n: int = 10) -> pd.DataFrame:
        """Plain filter on the real Category column - no scoring. See README for why category search was dropped."""
        subset = self.jobs_df[self.jobs_df["Category"] == category].copy()
        subset = subset.sort_values("word_count", ascending=False)

        deduped_rows = []
        seen = set()
        for _, row in subset.iterrows():
            key = (row["Title"], row["Company"])
            if key in seen:
                continue
            seen.add(key)
            locations = subset[
                (subset["Title"] == row["Title"]) & (subset["Company"] == row["Company"])
            ]["Location"].unique().tolist()
            row = row.copy()
            row["locations"] = locations
            deduped_rows.append(row)
            if len(deduped_rows) >= top_n:
                break

        return pd.DataFrame(deduped_rows) if deduped_rows else subset.head(0)

    def evidence_sentence(self, job_index: int, user_text: str) -> str:
        """Returns the single real sentence from the posting that overlaps most with the query - proof, not just a score."""
        text = self.jobs_df.loc[job_index, "full_text"]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 4]
        if not sentences:
            return ""

        user_terms = set(re.findall(r"[a-zA-Z]{3,}", user_text.lower()))
        if not user_terms:
            return sentences[0]

        best_sentence, best_overlap = sentences[0], -1
        for s in sentences:
            s_terms = set(re.findall(r"[a-zA-Z]{3,}", s.lower()))
            overlap = len(user_terms & s_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = s

        return best_sentence

    def trending_categories(self, top_n: int = 8) -> pd.DataFrame:
        return (
            self.jobs_df.groupby("Category")
            .size()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index(name="posting_count")
        )
