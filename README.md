# Resume-Analyser

This project consists of three different applications to analyze multiple résumés and determine which résumé is best suited for a given internship/job description. The applications use different approaches for keyword matching and semantic similarity: 

1. **app_scikit.py**: Uses TF-IDF Vectorizer and cosine similarity from `scikit-learn`.
2. **app_torch.py**: Utilizes `DistilBERT` from Hugging Face's `transformers` library for semantic embedding and matching.
3. **app_spaCy.py**: Leverages `spaCy` for semantic similarity and keyword matching.

## Features
- Upload multiple résumés (PDF format).
- Input job/internship descriptions for matching.
- Combines keyword matching and semantic similarity for accurate résumé analysis.
- Extracts and displays the best-matched candidate’s name and email.
- Allows downloading the best-matched résumé directly from the interface.

## Prerequisites

Ensure you have Python 3.x installed on your system. You can install all necessary dependencies from the `requirements.txt` file.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Dhakshayan/Resume-Analyser.git
   cd "Resume-Analyser"
