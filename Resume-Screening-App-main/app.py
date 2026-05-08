import streamlit as st
import pickle
import docx
import PyPDF2
import re

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Resume Screening app",
    page_icon="🚀",
    layout="wide"
)

# =========================================================
# LOAD MODEL FILES
# =========================================================

model = pickle.load(open("clf.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.lower()

# =========================================================
# PDF READER
# =========================================================

def extract_pdf(file):

    text = ""

    pdf_reader = PyPDF2.PdfReader(file)

    for page in pdf_reader.pages:

        content = page.extract_text()

        if content:
            text += content

    return text

# =========================================================
# DOCX READER
# =========================================================

def extract_docx(file):

    text = ""

    doc = docx.Document(file)

    for para in doc.paragraphs:

        text += para.text + "\n"

    return text

# =========================================================
# TXT READER
# =========================================================

def extract_txt(file):

    return file.read().decode("utf-8")

# =========================================================
# FILE HANDLER
# =========================================================

def get_resume_text(uploaded_file):

    extension = uploaded_file.name.split(".")[-1].lower()

    if extension == "pdf":
        return extract_pdf(uploaded_file)

    elif extension == "docx":
        return extract_docx(uploaded_file)

    elif extension == "txt":
        return extract_txt(uploaded_file)

    else:
        return ""

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_category(resume_text):

    cleaned = clean_text(resume_text)

    vectorized = tfidf.transform([cleaned])

    prediction = model.predict(vectorized)

    category = encoder.inverse_transform(prediction)

    return category[0]

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: #1e3a8a;
}

.subtitle {
    text-align: center;
    font-size: 24px;
    color: #475569;
}

.result-box {
    background: linear-gradient(to right,#2563eb,#4f46e5);
    color: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    font-size: 35px;
    font-weight: bold;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# UI
# =========================================================

st.markdown(
    "<div class='title'>🚀 Resume Screening app </div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Upload resume </div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx", "txt"]
)

# =========================================================
# PROCESS FILE
# =========================================================

if uploaded_file is not None:

    try:

        resume_text = get_resume_text(uploaded_file)

        st.success("Resume uploaded successfully!")

        if st.checkbox("Show Resume Text"):

            st.text_area(
                "Resume Content",
                resume_text,
                height=300
            )

        if st.button("Predict Category"):

            category = predict_category(resume_text)

            st.markdown(
                f"""
                <div class='result-box'>
                    {category}
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:

        st.error(f"Error: {e}")