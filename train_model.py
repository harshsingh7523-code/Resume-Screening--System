import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

data = {
    "resume": [
        "python pandas numpy machine learning data analysis",
        "machine learning ai model tensorflow sklearn",
        "html css javascript web development frontend",
        "react js frontend developer ui ux design",
        "java spring backend developer rest api",
        "node js express backend server development",
        "android app development kotlin java mobile",
        "flutter mobile app developer android ios"
    ],
    "category": [
        "Data Scientist",
        "AI ML ENGINEER",
        "Web Developer",
        "UI UX DESIGNER",
        "Backend Developer",
        "Technical Engineer",
        "Android Developer",
        "Software Developer"
    ]
}

df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['resume'])
y = df['category']

model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer,'vectorizer.pkl')
print('Logistic Regression model trained succesfull')