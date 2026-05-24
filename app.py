import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening System")

skills_list = [
    "python", "nlp", "machine learning", "tf-idf",
    "cosine similarity", "streamlit", "pandas",
    "sql", "django", "flask", "git", "power bi"
]

job_roles = {
    "Python Developer": "Python Django Flask SQL Git REST API Pandas",
    "NLP Developer": (
        "Python NLP Machine Learning TF-IDF Cosine Similarity "
        "NLTK spaCy Scikit-learn"
    ),
    "Data Analyst": "Python SQL Excel Pandas Power BI Statistics"
}


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text


def detect_skills(resume_text):
    resume_text = resume_text.lower()
    detected_skills = []

    for skill in skills_list:
        if skill in resume_text:
            detected_skills.append(skill.title())

    return detected_skills


def calculate_match_percentage(resume_text, job_description):
    documents = [resume_text, job_description]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(score * 100, 2)


st.title("AI Resume Screening System")
st.write("Upload your resume and check how well it matches a job role.")

selected_role = st.selectbox("Select Job Role", list(job_roles.keys()))
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)

    if resume_text.strip():
        detected_skills = detect_skills(resume_text)
        match_percentage = calculate_match_percentage(
            resume_text,
            job_roles[selected_role]
        )

        st.subheader("Results")
        st.write("Selected Role:", selected_role)
        st.metric("Match Percentage", f"{match_percentage}%")

        st.write("Detected Skills:")
        if detected_skills:
            st.success(", ".join(detected_skills))
        else:
            st.warning("No matching skills detected.")

        with st.expander("View Extracted Resume Text"):
            st.write(resume_text)
    else:
        st.error("Could not read text from this PDF.")