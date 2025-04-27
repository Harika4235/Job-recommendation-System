### job.py ###
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

# Check if the resources are already downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Cleaning function
def cleaning(txt):
    txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt)  # Remove special characters
    tokens = nltk.word_tokenize(txt.lower())  # Tokenize and lowercase
    stemming = [ps.stem(w) for w in tokens if w not in stopwords.words('english')]  # Remove stopwords and stem
    return " ".join(stemming)

# Load and prepare data
def load_and_prepare_data(filepath):
    job_df = pd.read_csv(filepath)
    job_df = job_df[['Job Title', 'Required Skills', 'Experience Level', 'Industry', 'Job Description']]
    job_df.fillna('', inplace=True)
    job_df = job_df.sample(n=5000, random_state=42)

    # Apply cleaning
    job_df['Job Description'] = job_df['Job Description'].astype(str).apply(cleaning)
    job_df['Job Title'] = job_df['Job Title'].astype(str).apply(cleaning)
    job_df['Required Skills'] = job_df['Required Skills'].astype(str).apply(cleaning)

    # Combine text fields for weighted features
    job_df['weighted_text'] = (
        job_df['Required Skills'] + " " + 
        job_df['Job Title'] + " " + 
        job_df['Job Description']
    )

    return job_df

# Compute similarity
def compute_similarity(job_df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(job_df['weighted_text'])
    return tfidf, matrix

# Evaluate recommendations
def evaluate_recommendations(recommended_jobs, actual_jobs, job_df):
    y_true = [1 if job in actual_jobs else 0 for job in job_df['Job Title']]
    y_pred = [1 if job in recommended_jobs['Job Title'].values else 0 for job in job_df['Job Title']]
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))


# Recommend jobs
def recommend_jobs(skills, job_df, tfidf, matrix):
    # Clean the user input
    skills_cleaned = cleaning(skills)
    user_vector = tfidf.transform([skills_cleaned])  # Transform user input into a vector

    # Compute similarity scores
    similarity_scores = cosine_similarity(user_vector, matrix).flatten()

    # Create a DataFrame with similarity scores
    job_df['similarity_score'] = similarity_scores

    # Sort by similarity scores in descending order
    sorted_jobs = job_df.sort_values(by='similarity_score', ascending=False)

    # Remove duplicate job titles (keeping the most relevant ones)
    unique_jobs = sorted_jobs.drop_duplicates(subset=['Job Title'])

    # Filter jobs with a similarity score above a certain threshold (e.g., 0.1)
    relevant_jobs = unique_jobs[unique_jobs['similarity_score'] > 0.1]

    # Select top 10 jobs
    top_jobs = relevant_jobs.head(10)

    # Return the most relevant fields
    return top_jobs[['Job Title', 'Required Skills', 'similarity_score']]