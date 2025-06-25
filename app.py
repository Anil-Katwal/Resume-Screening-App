# you need to install all these in your terminal
# pip install streamlit
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2


import streamlit as st
import pickle
import re
import pandas as pd
import plotly.express as px
import base64
import time
import os

# Try to import optional dependencies with error handling
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("⚠ python-docx not available. DOCX files will not be supported.")

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning(" PyPDF2 not available. PDF files will not be supported.")

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        svc_model = pickle.load(open('models/clf.pkl', 'rb'))
        tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
        le = pickle.load(open('models/encoder.pkl', 'rb'))
        return svc_model, tfidf, le
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

svc_model, tfidf, le = load_models()

# --- Helper Functions ---
def clean_resume(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(file):
    if not PDF_AVAILABLE:
        raise Exception("PDF support not available. Please install PyPDF2.")
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file):
    if not DOCX_AVAILABLE:
        raise Exception("DOCX support not available. Please install python-docx.")
    try:
        doc = docx.Document(file)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")

def extract_text_from_txt(file):
    try:
        file.seek(0)
        try:
            text = file.read().decode('utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            text = file.read().decode('latin-1')
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading TXT file: {str(e)}")

def handle_file_upload(uploaded_file):
    if uploaded_file is None:
        raise ValueError("No file uploaded")
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if uploaded_file.size > 10 * 1024 * 1024:
        raise ValueError("File size too large. Please upload a file smaller than 10MB.")
    if uploaded_file.size == 0:
        raise ValueError("File is empty")
    if file_extension == 'pdf':
        if not PDF_AVAILABLE:
            raise ValueError("PDF support not available. Please upload a TXT file instead.")
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        if not DOCX_AVAILABLE:
            raise ValueError("DOCX support not available. Please upload a TXT file instead.")
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Please upload PDF, DOCX, or TXT files.")
    if len(text.strip()) < 50:
        raise ValueError("Extracted text is too short. Please ensure the file contains readable text.")
    return text

def predict_category(resume_text):
    if svc_model is None or tfidf is None or le is None:
        return None, 0.0
    cleaned = clean_resume(resume_text)
    if len(cleaned) < 10:
        return None, 0.0
    vectorized = tfidf.transform([cleaned])
    pred = svc_model.predict(vectorized)[0]
    category = le.inverse_transform([pred])[0]
    confidence = 0.0
    if hasattr(svc_model, 'predict_proba'):
        proba = svc_model.predict_proba(vectorized)[0]
        confidence = max(proba) * 100
    return category, confidence

def get_download_link(data, filename, text):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# --- Streamlit App Layout ---
st.set_page_config(page_title="Resume Screening Web app", page_icon="🧑‍💼", layout="wide")

# Custom CSS for modern look
st.markdown("""
    <style>
    .main-title {font-size:2.8rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:0.5em;}
    .subtitle {font-size:1.2rem;color:#444;text-align:center;margin-bottom:2em;}
    .card {background:#f8f9fa;border-radius:12px;padding:2em 2em 1em 2em;box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:2em;}
    .result-card {background:#e3f2fd;border-radius:10px;padding:1.5em;margin-top:1em;box-shadow:0 2px 8px rgba(33,150,243,0.07);}
    .footer {text-align:center;color:#888;font-size:0.95em;margin-top:2em;}
    .stButton>button {background:#1976d2;color:white;font-weight:600;border-radius:8px;}
    .stDownloadButton>button {background:#43a047;color:white;font-weight:600;border-radius:8px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title"> 🧑‍💼Resume Screening AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a resume (PDF, DOCX, or TXT) and get an instant job category prediction powered by AI.</div>', unsafe_allow_html=True)

# Check if models are loaded
if svc_model is None:
    st.error(" Models could not be loaded. Please check if all model files are present.")
    st.stop()

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/ios-filled/100/1f77b4/resume.png", width=80)
st.sidebar.header("About this App")
st.sidebar.info(
    "This app uses a machine learning model to predict the job category of a resume. "
    "Upload a file and see the prediction instantly!\n\n"
    "**Supported formats:** PDF, DOCX, TXT\n"
    "**Max size:** 10MB"
)

# Show supported formats based on available dependencies
supported_formats = ["txt"]
if PDF_AVAILABLE:
    supported_formats.append("pdf")
if DOCX_AVAILABLE:
    supported_formats.append("docx")

st.sidebar.markdown("---")
st.sidebar.write("**How it works:**")
st.sidebar.write("1. Upload your resume file.")
st.sidebar.write("2. The app extracts and cleans the text.")
st.sidebar.write("3. The AI model predicts the job category.")
st.sidebar.write("4. See the result and download the summary.")

# --- Main Area ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Resume", type=supported_formats, help=f"Supported formats: {', '.join(supported_formats).upper()}")
    if uploaded_file:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                resume_text = handle_file_upload(uploaded_file)
                st.success("Text extracted successfully!")
                if st.checkbox("Show extracted text", False):
                    st.text_area("Extracted Resume Text", resume_text, height=250)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("Prediction Result")
                category, confidence = predict_category(resume_text)
                if category:
                    st.markdown(f"<h3 style='color:#1976d2;'>Predicted Category: <b>{category}</b></h3>", unsafe_allow_html=True)
                    st.markdown(f"<b>Confidence:</b> {confidence:.1f}%")
                else:
                    st.error("Could not predict a category. The resume text may be too short or invalid.")
                st.markdown('</div>', unsafe_allow_html=True)
                # Download result
                result_df = pd.DataFrame([{ 'Filename': uploaded_file.name, 'Category': category, 'Confidence': f"{confidence:.1f}%" }])
                csv = result_df.to_csv(index=False)
                st.markdown(get_download_link(csv.encode(), f"prediction_{uploaded_file.name}.csv", " Download Prediction as CSV"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Summary Section ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("How Accurate is the Model?")
try:
    with open('models/model_metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
    st.write(f"**Model:** {meta.get('model_name','-')}")
    st.write(f"**Accuracy:** {meta.get('accuracy',0):.2%}")
    st.write(f"**F1 Score:** {meta.get('f1_score',0):.2%}")
    st.write(f"**Categories:** {', '.join(meta.get('categories', []))}")
except Exception:
    st.info("Model metadata not found. Retrain the model for more info.")
st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">Made by Anil Katwal for Resume Screening | Powered by Streamlit & scikit-learn</div>', unsafe_allow_html=True)
