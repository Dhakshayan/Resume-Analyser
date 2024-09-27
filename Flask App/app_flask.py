from flask import Flask, render_template, request, send_file
import io
from pdfminer.high_level import extract_text
import spacy
from collections import Counter
import base64
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set your secret key

nlp = spacy.load("en_core_web_md")


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_file_data = pdf_file.read()  # Read the PDF content into bytes
    pdf_file_io = io.BytesIO(pdf_file_data)  # Create a file-like object

    return extract_text(pdf_file_io)


# Function to tokenize words
def word_tokenize(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc if not token.is_punct and not token.is_space]


# Function to compute keyword match
def compute_keyword_match(text, keywords):
    words = word_tokenize(text)
    stop_words = set(spacy.lang.en.stop_words.STOP_WORDS)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    word_count = Counter(filtered_words)
    keyword_count = sum(word_count[keyword] for keyword in keywords if keyword in word_count)

    return keyword_count


# Function to calculate match percentage between resume and job description
def calculate_percentage_match(text, job_description):
    job_keywords = set(word_tokenize(job_description.lower()))
    text_keywords_count = compute_keyword_match(text, job_keywords)
    total_keywords = len(job_keywords)

    if total_keywords == 0:
        return 0
    return (text_keywords_count / total_keywords) * 100


# Function to calculate semantic similarity between resume and job description
def semantic_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)


# Function to extract the first email found in the text
def extract_email(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    email_matches = re.findall(email_pattern, text)
    return email_matches[0] if email_matches else None


# Function to extract candidate name from resume text
def extract_candidate_name(text):
    lines = text.splitlines()
    common_generic_words = ['resume', 'curriculum vitae', 'cv']

    for line in lines[:10]:  # Check the first 10 lines for the name
        clean_line = line.strip()
        if clean_line.lower() not in common_generic_words and len(clean_line.split()) > 1:
            return clean_line
    return None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resumes = request.files.getlist('resumes')  # Get multiple uploaded resumes
        job_description = request.form['job_description']

        if resumes and job_description:
            best_resume_pdf = None
            best_match_score = -1
            best_resume_name = None
            best_resume_text = None
            best_candidate_email = None
            best_candidate_name = None

            for resume in resumes:
                resume_text = extract_text_from_pdf(resume)  # Extract text from the resume
                match_percentage = calculate_percentage_match(resume_text, job_description)
                semantic_sim = semantic_similarity(resume_text, job_description)
                match_score = (match_percentage + semantic_sim * 100) / 2

                # If this resume has a better match score, store its details
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_resume_pdf = resume.read()
                    best_resume_name = resume.filename
                    best_resume_text = resume_text
                    best_candidate_email = extract_email(resume_text)
                    best_candidate_name = extract_candidate_name(resume_text)

            # Pass the best-matched resume details to the result page
            return render_template('result.html',
                                   candidate_name=best_candidate_name,
                                   match_score=best_match_score,
                                   candidate_email=best_candidate_email,
                                   best_resume_name=best_resume_name)

    return render_template('index.html')


# Route to download the best-matched resume
@app.route('/download_resume')
def download_resume():
    resume_name = request.args.get('filename')
    resume_pdf = request.args.get('resume_pdf', type=str)

    if resume_name and resume_pdf:
        resume_pdf_bytes = io.BytesIO(base64.b64decode(resume_pdf))
        return send_file(resume_pdf_bytes, download_name=resume_name, as_attachment=True)
    return "No resume available for download", 404


if __name__ == '__main__':
    app.run(debug=True)
