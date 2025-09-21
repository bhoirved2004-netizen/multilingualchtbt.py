
import streamlit as st
import pandas as pd
import numpy as np
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Multilingual College Chatbot", layout="wide", page_icon="")

@st.cache_data
def load_data(path="D:\python\college_chatbot_dataset_500.csv"):
    return pd.read_csv("D:\python\college_chatbot_dataset_500.csv")

df = load_data()

# Sidebar controls
st.sidebar.title("Model & Dataset Controls")
classifier_choice = st.sidebar.selectbox("Classifier", ["LogisticRegression", "SVM (linear)"])
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
show_examples = st.sidebar.checkbox("Show sample Q&A", value=True)

st.title(" Multilingual College Chatbot (TF-IDF + Classifier)")
st.markdown("Multilingual Q&A (English, Hindi, Marathi) for college services. This demo uses TF-IDF features and a simple classifier to predict intent and serve answers in the detected language.")

if show_examples:
    st.subheader("Sample dataset overview")
    st.dataframe(df.sample(10).reset_index(drop=True))

# Prepare training data: combine multilingual questions, replicate intents accordingly
questions = pd.concat([df['question_en'], df['question_hi'], df['question_mr']], ignore_index=True)
intents = pd.concat([df['intent'], df['intent'], df['intent']], ignore_index=True)

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = vectorizer.fit_transform(questions)
y = intents

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size/100), random_state=int(random_state), stratify=y)

if classifier_choice == "LogisticRegression":
    model = LogisticRegression(max_iter=1000)
else:
    model = SVC(kernel='linear', probability=True)

with st.spinner("Training the classifier..."):
    model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric("Test Accuracy", f"{acc:.3f}")

st.subheader("Classification report")
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({"precision": "{:.2f}"}))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Chat interface
st.subheader("Chat with the bot")
user_input = st.text_input("Ask about admissions, exams, timetable, results or fees:", "")
if st.button("Send") or (user_input and st.session_state.get('auto_send', False)):
    query = user_input.strip()
    if query:
        # Detect language
        try:
            lang = detect(query)
        except:
            lang = "en"
        Xq = vectorizer.transform([query])
        pred_intent = model.predict(Xq)[0]
        # Pick most relevant answer (random sample among same-intent rows)
        cand = df[df['intent'] == pred_intent].sample(1).iloc[0]
        if str(lang).startswith('hi'):
            answer = cand['answer_hi']
        elif str(lang).startswith('mr'):
            answer = cand['answer_mr']
        else:
            answer = cand['answer_en']
        st.markdown(f"**Predicted intent:** `{pred_intent}`")
        st.markdown(f"**Bot:** {answer}")

# Download dataset
with open("D:\python\college_chatbot_dataset_500.csv","rb") as f:
    data = f.read()
st.sidebar.download_button("Download dataset (CSV)", data, file_name="college_chatbot_dataset_500.csv", mime='text/csv')

