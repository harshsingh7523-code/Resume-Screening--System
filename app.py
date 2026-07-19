import streamlit as st
import joblib
import numpy as np
import re
from PyPDF2 import PdfReader
import docx

model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")
le = joblib.load("label_encoder.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_text(file):
    text = ""

    if file.type == "application/pdf":
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() + "\n"
            return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + " "

    else:
        text = str(file.read(), "utf-8")

    return text


SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "php", "ruby", "swift",
    "html", "css", "react", "angular", "vue", "nodejs", "express", "django", "flask", "fastapi",
    "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis", "firebase",
    "machine learning", "deep learning", "artificial intelligence", "nlp",
    "data science", "data analysis", "pandas", "numpy", "scikit-learn", "tensorflow", "keras", "pytorch", "opencv",
    "power bi", "tableau", "matplotlib", "seaborn", "excel",
    "aws", "azure", "google cloud", "docker", "kubernetes", "linux","git", "github", "jenkins", "ci/cd",
    "vs code", "jupyter", "anaconda", "postman", "swagger","android", "flutter", "kotlin", "react native",
    "computer networks", "cyber security", "ethical hacking", "penetration testing",
    "hadoop", "spark", "kafka", "airflow", "etl", "statistics", "probability", "linear algebra",
    "business analysis", "financial analysis", "marketing analytics","communication", "teamwork", "leadership", "problem solving",
    "time management", "critical thinking", "adaptability", "rest api", "graphql", "microservices", "blockchain",
    "chatgpt", "prompt engineering", "automation", "Python", "Flask", "Django", "REST API", "Backend development", "Database integration", "Git",
     "Spring Boot", "Hibernate", "REST API", "Microservices", "OOPs concepts", "MySQL", "PostgreSQL", "MongoDB", "Database optimization", "backup", "recovery",
    "Artificial Intelligence", "Deep Learning", "Neural networks", "NLP", "Figma", "Adobe XD", "UI design", "wireframes", "user experience", "Troubleshooting", "customer support", "networking", "system maintenance"
    ]


def extract_skills(text):
    text = text.lower()
    found = [skill for skill in SKILLS if skill in text]
    return list(set(found))

st.title(" Resume Screening System")
st.write("Upload your resume to get Category Prediction and Skills")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    text = extract_text(uploaded_file)

    if text.strip() == "":
        st.error("Could not read text from file!")
    else:
        st.subheader("Extracted Text")
        st.write(text[:1000])

        cleaned = clean_text(text)

        vector = tfidf.transform([cleaned])

        # Prediction
        prediction = model.predict(vector)
        category = le.inverse_transform(prediction)[0]
        
        skills = extract_skills(text)
        
        st.success(f"Predicted Category: {category}")
        st.warning(f"Skills Found: {', '.join(skills) if skills else 'None'}")