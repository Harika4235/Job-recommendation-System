import streamlit as st
from job import load_and_prepare_data, compute_similarity, recommend_jobs

# Streamlit Frontend
st.title("Job Recommendation System")
st.write("Enter your skills to find matching jobs.")

# Path to the dataset
data_path = "C:\\Users\\HARIKA\\OneDrive\\Desktop\\skill-job\\detailed_job_dataset.csv"

# Load and prepare data
job_df = load_and_prepare_data(data_path)
tfidf, matrix = compute_similarity(job_df)

# Input field
skills = st.text_input("Enter your skills (comma-separated):")

if st.button("Find Jobs"):
    if skills.strip():
        recommended_jobs = recommend_jobs(skills, job_df, tfidf, matrix)
        st.write("Recommended Jobs:")
        st.dataframe(recommended_jobs)
    else:
        st.error("Please enter at least one skill.")
