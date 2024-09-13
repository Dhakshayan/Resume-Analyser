

#using scikit-learn

import streamlit as st
import io
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import base64
import re

def extract_text_from_pdf(pdf_file):
    return extract_text(pdf_file)

def compute_keyword_match(text, keywords):
    word_count = Counter(text.lower().split())
    keyword_count = sum(word_count[keyword] for keyword in keywords if keyword in word_count)
    return keyword_count


def calculate_percentage_match(text, job_description):
    job_keywords = set(job_description.lower().split())
    text_keywords_count = compute_keyword_match(text, job_keywords)
    total_keywords = len(job_keywords)

    if total_keywords == 0:
        return 0
    return (text_keywords_count / total_keywords) * 100


def semantic_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0] * 100


def extract_email(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    email_matches = re.findall(email_pattern, text)
    return email_matches[0] if email_matches else None


def extract_candidate_name(text):
    lines = text.splitlines()
    common_generic_words = ['resume', 'curriculum vitae', 'cv']

    for line in lines[:10]:
        clean_line = line.strip()
        if clean_line.lower() not in common_generic_words and len(clean_line.split()) > 1:
            return clean_line
    return None


st.set_page_config(page_title="Résumé Match Predictor", layout="wide")
st.markdown('<h1 style="color: #1E90FF;">Résumé-Internship Match Predictor</h1>', unsafe_allow_html=True)


st.markdown("""<p style="color:#1E90FF; font-size:20px;">
    <strong>Welcome!</strong><br>
    Upload multiple résumés and enter the job description to see which résumé matches best!
    </p>
    """,unsafe_allow_html=True
)

uploaded_files = st.file_uploader("Upload Résumés (PDF)", type="pdf", accept_multiple_files=True)

job_description = st.text_area("Enter Job Description (do not use full sentences; only key points)", height=150)

if st.button('Analyze Résumés'):
    if uploaded_files and job_description:
        with st.spinner('Processing...'):
            best_resume_pdf = None
            best_match_score = -1
            best_resume_name = None
            best_resume_text = None
            best_candidate_email = None
            best_candidate_name = None

            for uploaded_file in uploaded_files:
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if not resume_text.strip():
                    st.warning(f"No text extracted from {uploaded_file.name}. It may be an image-based PDF.")
                    continue

                match_percentage = calculate_percentage_match(resume_text, job_description)
                semantic_sim = semantic_similarity(resume_text, job_description)
                match_score = (match_percentage + semantic_sim) / 2

                if match_score > best_match_score:
                    best_match_score = match_score
                    best_resume_pdf = uploaded_file.getvalue()
                    best_resume_name = uploaded_file.name
                    best_resume_text = resume_text
                    best_candidate_email = extract_email(resume_text)
                    best_candidate_name = extract_candidate_name(resume_text)

            st.markdown("### Best Matched Résumé:")
            st.write(f"**{best_candidate_name}** is best suited for this internship with a combined match score of {best_match_score:.2f}%")
            
            if best_candidate_email:
                st.write(f"Email ID: {best_candidate_email}")
            else:
                st.write("Email ID: Not found")

            base64_pdf = base64.b64encode(best_resume_pdf).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1000" height="900" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.download_button("Download Best Matched Résumé", data=best_resume_pdf, file_name=best_resume_name)
    else:
        st.error("Please upload résumés and enter job descriptions.")

