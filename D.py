import streamlit as st
import pdfplumber
import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def preprocess_text(text):
    """Cleans and preprocesses text using NLP."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def rank_resumes(resume_texts, job_description):
    """Ranks resumes based on relevance to the job description using TF-IDF & cosine similarity."""
    vectorizer = TfidfVectorizer()
    corpus = [preprocess_text(job_description)] + [preprocess_text(resume) for resume in resume_texts]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
    return similarity_scores
def main():
    st.title("AI-powered Resume Screening and Ranking System")
    st.write("Upload resumes and provide a job description to rank candidates.")
    
    # Job Description Input
    job_description = st.text_area("Enter the Job Description:")
    
    # Resume Uploads
    uploaded_files = st.file_uploader("Upload Resumes (PDFs)", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Rank Resumes") and job_description and uploaded_files:
        resume_texts = [extract_text_from_pdf(file) for file in uploaded_files]
        scores = rank_resumes(resume_texts, job_description)
        
        # Display Results
        ranked_resumes = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)
        results = pd.DataFrame({
            "Candidate": [res[0].name for res in ranked_resumes],
            "Match Score": [round(res[1] * 100, 2) for res in ranked_resumes]
        })
        st.dataframe(results)

if __name__ == "__main__":
    main()
