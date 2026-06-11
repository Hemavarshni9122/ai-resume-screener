# 🤖 AI-Powered Resume Screening & Candidate Ranking System

An NLP-based machine learning web application that automatically analyses and ranks resumes based on a given job description using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 📌 Project Overview

Recruiters spend hours manually reviewing resumes. This system automates the screening process by:
- Converting resumes and job descriptions into numerical vectors using TF-IDF
- Calculating the similarity score between each resume and the job description
- Ranking candidates from best match to lowest
- Highlighting the key matching keywords for top candidates

---

## 🧠 Concepts Used

### 1. Text Preprocessing
- Lowercasing, punctuation removal, whitespace normalisation
- Removes noise so the ML model focuses on meaningful words

### 2. TF-IDF (Term Frequency – Inverse Document Frequency)
- **TF**: How often a word appears in a document
- **IDF**: How rare/important that word is across all documents
- Result: important domain-specific words (e.g. "machine learning") get higher scores than common words (e.g. "the", "and")

### 3. Cosine Similarity
- Measures the angle between two document vectors
- Score of 1.0 = identical, Score of 0.0 = nothing in common
- Used to compare each resume vector against the job description vector

### 4. Keyword Extraction
- Identifies which key terms from the job description appear in each resume
- Explains WHY a candidate ranked high or low

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Scikit-learn | TF-IDF Vectorizer, Cosine Similarity |
| Pandas | Data handling and results formatting |
| Streamlit | Web application UI |
| Regex (re) | Text preprocessing |

---

## 📁 Project Structure

```
resume_screener/
│
├── app.py            # Streamlit web app (frontend + UI logic)
├── ranker.py         # ML core (preprocessing, TF-IDF, ranking)
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## 🚀 How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the app
streamlit run app.py
```

Then open your browser at: `http://localhost:8501`

---

## 📊 How to Use

1. Paste a **job description** in the left panel
2. Add candidate names and their **resume text**
3. Click **Screen & Rank Resumes**
4. View ranked results with match scores and keyword highlights
5. Download results as **CSV**

---

## 📈 Sample Output

| Rank | Candidate | Match % |
|---|---|---|
| 🥇 1 | Hemavarshni S | 78.4% |
| 🥈 2 | Candidate B | 61.2% |
| 🥉 3 | Candidate C | 43.7% |

---

## 💡 Future Improvements

- PDF resume upload support (using PyPDF2)
- Support for DOCX files (using python-docx)
- Transformer-based similarity (BERT/Sentence-BERT) for deeper semantic matching
- Multi-language support
- Database storage for candidate history

---

## 👩‍💻 Author

**Hemavarshni S**  
B.Tech – Artificial Intelligence & Data Science  
Loyola-ICAM College of Engineering & Technology, Chennai
