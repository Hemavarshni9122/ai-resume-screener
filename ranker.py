# ============================================================
# ranker.py — The Brain of the Project
# This file handles all the NLP and ML logic.
# Streamlit just calls functions from this file.
# ============================================================

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------
# STEP 1: TEXT PREPROCESSING
# Why? Raw text has noise — punctuation, UPPERCASE, 
# stopwords like "the/and/is". We clean it so TF-IDF 
# works on meaningful words only.
# -------------------------------------------------------

def preprocess(text: str) -> str:
    """
    Cleans a block of text:
    - Lowercases everything       → 'Python' and 'python' become same
    - Removes punctuation/numbers → only words remain
    - Strips extra spaces         → tidy output
    """
    text = text.lower()                        # lowercase
    text = re.sub(r'[^a-z\s]', ' ', text)     # remove non-alphabet chars
    text = re.sub(r'\s+', ' ', text).strip()  # collapse multiple spaces
    return text


# -------------------------------------------------------
# STEP 2: TF-IDF VECTORIZATION
# Why? Computers can't understand text — only numbers.
# TF-IDF converts each document into a vector of numbers.
#
# TF  = Term Frequency    → how often a word appears in THIS doc
# IDF = Inverse Doc Freq  → how rare the word is across ALL docs
# 
# Result: common words like "the" get low scores.
#         important words like "machine learning" get high scores.
# -------------------------------------------------------

def vectorize(documents: list) -> tuple:
    """
    Takes a list of text documents.
    Returns:
      - tfidf_matrix : numerical representation of all documents
      - vectorizer   : fitted TF-IDF object (reusable)
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',   # ignore common English words (the, is, at...)
        ngram_range=(1, 2),     # consider single words AND two-word phrases
                                # e.g. "machine learning" as one feature
        max_features=5000       # keep top 5000 most important features
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer


# -------------------------------------------------------
# STEP 3: COSINE SIMILARITY
# Why? We need to measure HOW SIMILAR each resume is 
# to the job description.
#
# Think of each document as an arrow in space.
# Cosine similarity measures the ANGLE between two arrows.
# Angle = 0°  → score = 1.0 (identical)
# Angle = 90° → score = 0.0 (nothing in common)
# -------------------------------------------------------

def rank_resumes(job_description: str, resumes: dict) -> pd.DataFrame:
    """
    Main function — ranks resumes against a job description.
    
    Parameters:
      job_description : string of the job posting text
      resumes         : dict of { "Candidate Name": "resume text", ... }
    
    Returns:
      A sorted DataFrame with columns: Rank, Candidate, Score, Match %
    """

    # --- Prepare documents ---
    # We put the job description FIRST, then all resumes after it.
    # This way index[0] = job description, index[1..n] = resumes
    candidate_names = list(resumes.keys())
    candidate_texts = list(resumes.values())

    # Clean all text
    cleaned_jd = preprocess(job_description)
    cleaned_resumes = [preprocess(text) for text in candidate_texts]

    # Combine into one list for vectorization
    all_documents = [cleaned_jd] + cleaned_resumes

    # --- TF-IDF Vectorization ---
    tfidf_matrix, _ = vectorize(all_documents)

    # --- Cosine Similarity ---
    # Compare job description (row 0) against all resumes (rows 1 to n)
    jd_vector = tfidf_matrix[0]          # job description vector
    resume_vectors = tfidf_matrix[1:]    # all resume vectors

    # cosine_similarity returns a 2D array, we flatten to 1D list
    similarity_scores = cosine_similarity(jd_vector, resume_vectors).flatten()

    # --- Build Results DataFrame ---
    results = pd.DataFrame({
        'Candidate': candidate_names,
        'Score': similarity_scores,
        'Match %': [f"{score * 100:.1f}%" for score in similarity_scores]
    })

    # Sort by score descending (best match first)
    results = results.sort_values('Score', ascending=False).reset_index(drop=True)
    results.insert(0, 'Rank', range(1, len(results) + 1))

    return results


# -------------------------------------------------------
# STEP 4: KEYWORD EXTRACTION
# Bonus feature — shows WHICH keywords from the job 
# description are present in each resume.
# Helps explain WHY a candidate ranked high or low.
# -------------------------------------------------------

def extract_keywords(job_description: str, resume_text: str, top_n: int = 10) -> list:
    """
    Finds the most important JD keywords present in a resume.
    Uses TF-IDF on just these two documents to find key terms.
    """
    cleaned_jd = preprocess(job_description)
    cleaned_resume = preprocess(resume_text)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform([cleaned_jd, cleaned_resume])
        feature_names = vectorizer.get_feature_names_out()

        # Get scores for the job description (index 0)
        jd_scores = matrix[0].toarray().flatten()

        # Sort by importance
        top_indices = jd_scores.argsort()[::-1][:top_n]
        keywords = [feature_names[i] for i in top_indices if jd_scores[i] > 0]
        return keywords
    except Exception:
        return []
