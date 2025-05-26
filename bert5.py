
# Corporate Clause Risk Analyzer - Streamlit Implementation with NLP
# Required packages:
# pip install streamlit pdfminer.six docx2txt nltk scikit-learn spacy pandas joblib transformers torch
# python -m spacy download en_core_web_sm
import tempfile
import os
import re
import nltk
import spacy
import docx2txt
import time
import numpy as np
import pandas as pd
import streamlit as st
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import sent_tokenize
import torch
from transformers import BertTokenizer, BertModel, pipeline
import plotly.express as px
import random
import string




# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="CORPORATE CLAUSE RISK ANALYZER",
    page_icon="📑",
    layout="wide"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        body {
            background-color: #F1F1F2;
            color: #1995AD;
        }
        .stApp {
             background: linear-gradient(to bottom right, #f1f9fb, #e4f2f6);
        }
        [data-testid="stSidebar"] {
            background-color: linear-gradient(135deg, #1c4b82, #57c1eb); !important; /* Darker turquoise background */
            color:#1995AD !important; /* White text for better contrast */
            border-left: 4px solid #05414c; /* Even darker border */
            font-family: 'Roboto', sans-serif !important;
            font-weight: 500; /* Medium weight for all sidebar text */
        }

        
        /* Apply Roboto to all sidebar elements */
        [data-testid="stSidebar"] .stMarkdown, 
        [data-testid="stSidebar"] .stButton>button,
        [data-testid="stSidebar"] .stTitle, 
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] .stTextInput>div>div>input,
        [data-testid="stSidebar"] .stSelectbox {
            font-family: 'Roboto', sans-serif !important;
        }
        
        /* Make sidebar headers more pronounced with Roboto */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        
        /* Enhancing sidebar info sections */
        [data-testid="stSidebar"] .stAlert {
            font-family: 'Roboto', sans-serif !important;
            border-radius: 8px;
        }
        
        /* Keep your existing styles below... */
        
        /* Text styling — exclude .stButton>button */
        .stMarkdown, .stTextInput, .stTextArea, .stRadio, 
        .stSelectbox, .stCheckbox, label, h1, h2, h3, h4, h5, h6, p, div, span {
            color: #1995AD !important;
        }
        .risk-box {
            background-color: #ffffff;
            border: 1px solid #1995AD;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }
        
        .expander-content {
            padding: 15px;
            margin-top: 5px;
        }
        
        /* Risk level indicators */
        .risk-indicator {
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 12px;
            display: inline-block;
            margin-left: 10px;
        }
        .high-risk {
            background-color: rgba(255, 0, 0, 0.1);
            color: #ff0000;
            border: 1px solid #ff0000;
        }
        .medium-risk {
            background-color: rgba(255, 140, 0, 0.1);
            color: #ff8c00;
            border: 1px solid #ff8c00;
        }
        .low-risk {
            background-color: rgba(0, 128, 0, 0.1);
            color: #008000;
            border: 1px solid #008000;
        }
        /* Button styling */
        
.stButton > button {
        background-color: black;
        color: white;
    }

    /* Target DISABLED Streamlit buttons */
    .stButton > button:disabled {
        color: white !important;
        opacity: 1 !important;
        background-color: black !important;
    }
        /* File uploader styling */
        [data-testid="stFileUploader"] {
            border-radius: 10px;
            padding: 1em;
            color: white;
        }
        [data-testid="stFileUploader"] section {
            color: white;
        }
 
        /* Sidebar info box override */
        [data-testid="stSidebar"] .stAlert {
            background-color: #ace2ef !important;
            color: #1788A0 !important;
            border-left: 4px solid #1995AD;
        }
        [data-testid="stSidebar"] .stAlert p {
            color: #1995AD !important;
         [data-testid="stSidebar"] h1 {
    position: sticky;
    top: 0;
    background-color: #ace2ef;
    padding: 8px;
    z-index: 10;
}
body, .stApp {
    font-family: 'Segoe UI', sans-serif !important;
}

    </style>
    """,
    unsafe_allow_html=True
)




# Initialize spaCy and other NLP components
@st.cache_resource
def load_nlp_resources():
    nlp = spacy.load("en_core_web_sm")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return nlp, bert_tokenizer, bert_model, summarizer

nlp, bert_tokenizer, bert_model, summarizer = load_nlp_resources()

# Define common corporate clause types
CLAUSE_TYPES = {
    'indemnification': [
        'indemnify', 'indemnification', 'hold harmless', 'defend', 'liability', 
        'claim', 'damage', 'loss', 'expense', 'attorney'
    ],
    'limitation_of_liability': [
        'limitation of liability', 'limited liability', 'cap on liability', 'maximum liability',
        'aggregate liability', 'shall not exceed', 'not be liable', 'direct damages'
    ],
    'termination': [
        'termination', 'terminate', 'notice period', 'breach', 'cure period', 
        'discontinue', 'cancellation', 'wind up', 'end of term'
    ],
    'confidentiality': [
        'confidential', 'confidentiality', 'non-disclosure', 'proprietary information',
        'trade secret', 'disclosure', 'protect', 'safeguard'
    ],
    'payment_terms': [
        'payment', 'invoice', 'net', 'days', 'fee', 'charge', 'rate', 'price',
        'discount', 'penalty', 'interest'
    ],
    'warranty': [
        'warranty', 'warrants', 'guarantee', 'represents', 'representation',
        'merchantability', 'fitness', 'satisfactory quality', 'as is'
    ],
    'intellectual_property': [
        'intellectual property', 'ip', 'copyright', 'patent', 'trademark',
        'license', 'ownership', 'proprietary right', 'work product'
    ],
    'governing_law': [
        'governing law', 'jurisdiction', 'venue', 'dispute', 'arbitration',
        'mediation', 'court', 'law of', 'governed by'
    ],
    'force_majeure': [
        'force majeure', 'act of god', 'beyond control', 'unforeseen',
        'unavoidable', 'pandemic', 'natural disaster', 'war', 'strike'
    ]
}

# Risk patterns for each risk level
RISK_PATTERNS = {
    'high': [
        r'unlimited liability', r'sole discretion', r'non-negotiable', r'shall indemnify.*all',
        r'broad.*indemnification', r'exclusive remedy', r'waive.*rights',
        r'disclaim.*warranties', r'no.*warranty', r'as is', r'perpetual', r'irrevocable',
        r'unilateral.*termination', r'forfeit', r'without.*notice', r'immediate termination',
        r'uncapped', r'consequential damages'
    ],
    'medium': [
        r'reasonable efforts', r'material breach', r'third party claims',
        r'notice.*cure', r'limited warranty', r'discretion', r'may terminate',
        r'non-refundable', r'\d+ days notice', r'renewal.*automatically', 
        r'limited to (direct|actual) damages', r'cap(ped)? at'
    ],
    'low': [
        r'best efforts', r'mutual', r'reasonable notice', r'consent not.*unreasonably withheld',
        r'reasonable time', r'good faith', r'commercially reasonable', r'jointly'
    ]
}

def extract_text_from_file(uploaded_file):
    """Extract text from PDF or DOCX files."""
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = temp.name
    
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.pdf':
            text = extract_text(temp_path)
        elif file_extension == '.docx':
            text = docx2txt.process(temp_path)
        elif file_extension == '.txt':
            text = uploaded_file.getvalue().decode('utf-8')
        else:
            text = ""
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)
    
    return text

def split_into_clauses(text):
    """Split the document into potential clauses using NLP."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Look for common clause indicators
    clause_markers = [
        r'\d+\.\d+\s+[A-Z]',  # "1.1 Term"
        r'[A-Z][A-Z\s]+\.',   # "WARRANTY."
        r'Section\s+\d+',     # "Section 7"
        r'ARTICLE\s+[IVX]+',  # "ARTICLE IV" 
        r'\d+\.\s+[A-Z]'      # "1. Definitions"
    ]
    
    # Create initial splits based on markers
    split_points = []
    for marker in clause_markers:
        for match in re.finditer(marker, text):
            split_points.append(match.start())
    
    # Sort split points
    split_points = sorted(set(split_points))
    
    # Create clauses
    clauses = []
    for i in range(len(split_points)):
        start = split_points[i]
        end = split_points[i+1] if i+1 < len(split_points) else len(text)
        clause_text = text[start:end].strip()
        if len(clause_text) > 50:  # Ignore very short segments
            clauses.append(clause_text)
    
    # If no clauses were found using markers, try splitting by paragraphs
    if not clauses:
        paragraphs = re.split(r'\n\s*\n', text)
        clauses = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    # Further process with spaCy to refine clauses
    if len(clauses) == 1 and len(clauses[0]) > 1000:
        # If there's just one large chunk, try to split it using sentence boundaries
        doc = nlp(clauses[0])
        sentences = list(doc.sents)
        
        # Group sentences into logical clauses (approximate method)
        refined_clauses = []
        current_clause = ""
        for sent in sentences:
            current_clause += sent.text + " "
            # If sentence ends with a period and next char is uppercase, or if it contains certain keywords
            if (sent.text.endswith('.') or any(keyword in sent.text.lower() for keyword in ['provided that', 'subject to', 'notwithstanding'])):
                if len(current_clause) > 50:
                    refined_clauses.append(current_clause.strip())
                    current_clause = ""
        
        if current_clause and len(current_clause) > 50:
            refined_clauses.append(current_clause.strip())
            
        if refined_clauses:
            clauses = refined_clauses
    
    return clauses

def get_bert_embeddings(text):
    """Get BERT embeddings for text for more sophisticated analysis"""
    # Truncate if text is too long (BERT has a limit of 512 tokens)
    encoded_input = bert_tokenizer(text, truncation=True, padding=True, 
                                  max_length=512, return_tensors='pt')
    
    with torch.no_grad():
        output = bert_model(**encoded_input)
    
    # Use the CLS token embedding as the sentence embedding
    sentence_embedding = output.last_hidden_state[:, 0, :].numpy()
    return sentence_embedding[0]  # Return as a 1D array

def classify_clause_type(clause_text):
    """Identify the type of clause using NLP analysis."""
    clause_text_lower = clause_text.lower()
    scores = {}
    
    # Basic keyword matching
    for clause_type, keywords in CLAUSE_TYPES.items():
        score = sum(1 for keyword in keywords if keyword.lower() in clause_text_lower)
        scores[clause_type] = score
    
    # Enhanced with spaCy NLP analysis
    doc = nlp(clause_text)
    
    # Check for entities that might indicate specific clause types
    for ent in doc.ents:
        if ent.label_ == "LAW" and "governing_law" in scores:
            scores["governing_law"] += 2
        elif ent.label_ == "ORG" and "confidentiality" in scores:
            scores["confidentiality"] += 1
    
    # Look for verb patterns that might indicate clause types
    for token in doc:
        if token.pos_ == "VERB":
            verb = token.text.lower()
            if verb in ["indemnify", "defend", "protect"]:
                scores["indemnification"] = scores.get("indemnification", 0) + 2
            elif verb in ["terminate", "cancel", "end"]:
                scores["termination"] = scores.get("termination", 0) + 2
            elif verb in ["pay", "reimburse", "charge"]:
                scores["payment_terms"] = scores.get("payment_terms", 0) + 2
    
    # Return the clause type with the highest score, if it's above 0
    max_score = max(scores.values()) if scores else 0
    if max_score > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    return "other"

def rate_clause_risk(clause_text, clause_type):
    """Rate the risk level of a clause as low, medium, or high using NLP analysis."""
    clause_text_lower = clause_text.lower()
    
    # Check for risk patterns
    high_risk_matches = sum(1 for pattern in RISK_PATTERNS['high'] 
                           if re.search(pattern, clause_text_lower))
    medium_risk_matches = sum(1 for pattern in RISK_PATTERNS['medium'] 
                             if re.search(pattern, clause_text_lower))
    low_risk_matches = sum(1 for pattern in RISK_PATTERNS['low'] 
                          if re.search(pattern, clause_text_lower))
    
    # Enhanced NLP analysis
    doc = nlp(clause_text)
    
    # Check for obligation language
    obligation_terms = ["shall", "must", "required", "obligation", "mandatory"]
    obligation_count = sum(1 for token in doc if token.text.lower() in obligation_terms)
    
    # Check for limiting language
    limiting_terms = ["limited", "reasonable", "solely", "only", "to the extent"]
    limiting_count = sum(1 for token in doc if token.text.lower() in limiting_terms)
    
    # Check for absolutes
    absolute_terms = ["all", "any", "every", "always", "never", "under no circumstance"]
    absolute_count = sum(1 for token in doc if token.text.lower() in absolute_terms)
    
    # Analyze negations (high risk might have more negations)
    negation_count = sum(1 for token in doc if token.dep_ == "neg")
    
    # Contextual analysis based on clause type
    type_risk_score = 0
    
    if clause_type == "indemnification":
        if "unlimited" in clause_text_lower or "all claims" in clause_text_lower:
            type_risk_score += 3  # High risk
        elif "third party" in clause_text_lower and "limited to" in clause_text_lower:
            type_risk_score += 2  # Medium risk
        elif "mutual" in clause_text_lower:
            type_risk_score += 1  # Low risk
    
    elif clause_type == "limitation_of_liability":
        if "no liability" in clause_text_lower or "in no event" in clause_text_lower:
            type_risk_score += 3
        elif "limited to fees paid" in clause_text_lower:
            type_risk_score += 2
        elif "mutual limitation" in clause_text_lower:
            type_risk_score += 1
    
    elif clause_type == "termination":
        if "immediate" in clause_text_lower and "notice" not in clause_text_lower:
            type_risk_score += 3
        elif "notice" in clause_text_lower and "cure" not in clause_text_lower:
            type_risk_score += 2
        elif "mutual" in clause_text_lower and "notice" in clause_text_lower:
            type_risk_score += 1
    
    # Calculate final risk score
    risk_score = (high_risk_matches * 3) + (medium_risk_matches * 2) + (low_risk_matches * -1) + \
                (obligation_count * 0.5) + (limiting_count * -0.5) + (absolute_count * 1) + \
                (negation_count * 0.5) + type_risk_score
    
    # Determine risk level based on score
    if risk_score >= 3:
        return "high"
    elif risk_score >= 1:
        return "medium"
    else:
        return "low"

def summarize_clause(clause_text):
    """Generate a concise summary of the clause using NLP."""
    # For short clauses, just return the text
    if len(clause_text.split()) < 50:
        return clause_text
    
    try:
        # Try using the transformer-based summarizer for longer texts
        if len(clause_text.split()) > 50:
            # Truncate if too long for the model
            max_tokens = 1024
            words = clause_text.split()
            if len(words) > max_tokens:
                truncated_text = " ".join(words[:max_tokens])
            else:
                truncated_text = clause_text
                
            summary = summarizer(truncated_text, max_length=100, min_length=30, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error in transformer summarization: {e}")
        # Fall back to extractive summarization
    
    # Fallback: Simple extractive summarization
    doc = nlp(clause_text)
    sentences = [sent.text for sent in doc.sents]
    
    if len(sentences) <= 2:
        return sentences[0]
    
    # For longer clauses, try to extract the most representative sentences
    try:
        # Use TF-IDF to find most important sentences
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).tolist()
        top_idx = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:2]
        summary_sentences = [sentences[i] for i in sorted(top_idx)]
        return ' '.join(summary_sentences)
    except:
        # Fallback to first and last sentence if TF-IDF fails
        return f"{sentences[0]} ... {sentences[-1]}"

def analyze_document(text):
    """Analyze a document for clause risks using NLP."""
    if not text:
        return {"error": "No text to analyze"}
    

    
    clauses = split_into_clauses(text)
    results = []
    
    # Process each clause
    for clause_text in clauses:
        clause_type = classify_clause_type(clause_text)
        risk_level = rate_clause_risk(clause_text, clause_type)
        summary = summarize_clause(clause_text)
        
        results.append({
            "text": clause_text,
            "summary": summary,
            "type": clause_type,
            "risk_level": risk_level
        })
    
    return {
        "total_clauses": len(results),
        "high_risk_count": sum(1 for r in results if r["risk_level"] == "high"),
        "medium_risk_count": sum(1 for r in results if r["risk_level"] == "medium"),
        "low_risk_count": sum(1 for r in results if r["risk_level"] == "low"),
        "clauses": results
    }

def show_analysis_progress():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        # Simulating work
        status_text.text(f"Processing document: {i+1}%")
        progress_bar.progress(i + 1)
        time.sleep(0.01)  # Very short delay to not slow down actual processing
    
    status_text.text("Displaying results shortly!")
# --- START OF AUTH SECTION ---
import hashlib
import csv

USER_DB_PATH = "users.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

from datetime import datetime

def save_user(username, password):
    hashed = hash_password(password)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(USER_DB_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, hashed, timestamp])


def user_exists(username):
    if not os.path.exists(USER_DB_PATH):
        return False
    with open(USER_DB_PATH, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == username:
                return True
    return False

def authenticate_user(username, password):
    if not os.path.exists(USER_DB_PATH):
        return False
    hashed = hash_password(password)
    with open(USER_DB_PATH, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == username and row[1] == hashed:
                return True
    return False
def generate_captcha(length=5):
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choices(characters, k=length))

def login_or_signup():
    st.sidebar.title("Login / Sign Up")

    option = st.sidebar.radio("Choose an option", ["Login", "Sign Up"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if "captcha_code" not in st.session_state:
        st.session_state["captcha_code"] = generate_captcha()

    # Display CAPTCHA
    st.sidebar.markdown(f"**CAPTCHA:** `{st.session_state['captcha_code']}`")
    user_input = st.sidebar.text_input("Enter CAPTCHA exactly")

    if option == "Sign Up":
        if st.sidebar.button("Register"):
            if user_input != st.session_state["captcha_code"]:
                st.sidebar.error("❌ Incorrect CAPTCHA. Try again.")
                return

            if user_exists(username):
                st.sidebar.warning("Username already exists.")
            else:
                save_user(username, password)
                st.sidebar.success("✅ Registration successful. Please login.")
                del st.session_state["captcha_code"]
    else:
        if st.sidebar.button("Login"):
            if user_input != st.session_state["captcha_code"]:
                st.sidebar.error("❌ Incorrect CAPTCHA. Try again.")
                
                return

            if authenticate_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password.")
                
# --- END OF AUTH SECTION ---


# Streamlit UI
def main():
   # Gradient title with Aileron-like font
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;900&display=swap');
            
            .gradient-title {
                font-family: 'Poppins', sans-serif; /* Closest web-safe alternative to Aileron */
                font-weight: 900;
                font-size: 2.8rem;
                text-transform: uppercase;
                background: linear-gradient(135deg, #1995AD 0%, #0f5c6b 50%, #085066 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                color: transparent;
                text-shadow: 2px 2px 4px rgba(25, 149, 173, 0.2);
                letter-spacing: 0.5px;
                padding: 10px 0;
                position: relative;
                display: inline-block;
                margin-bottom: 5px;
            }
            
            .gradient-title:after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, transparent, #1995AD, transparent);
                border-radius: 2px;
            }
            
            .subtitle {
                font-family: 'Poppins', sans-serif;
                font-weight: 400;
                font-size: 1.2rem;
                color: #1995AD;
                margin-top: 0;
            }
            /* Updated Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #1995AD 0%, #0f5c6b 70%, #085066 100%) !important;
    padding: 1rem;
    color: white !important;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] .stMarkdown {
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
}

[data-testid="stSidebar"] h1 {
    font-weight: 700;
    font-size: 1.8rem;
    border-bottom: 2px solid rgba(255, 255, 255, 0.3);
    padding-bottom: 10px;
    margin-bottom: 15px;
}

[data-testid="stSidebar"] h3 {
    font-weight: 600;
    font-size: 1.2rem;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* Style for the list items in sidebar */
[data-testid="stSidebar"] ul {
    list-style-type: none;
    padding-left: 10px;
    margin-top: 0;
}

[data-testid="stSidebar"] li {
    position: relative;
    padding-left: 20px;
    margin-bottom: 8px;
    font-family: 'Poppins', sans-serif !important;
}

[data-testid="stSidebar"] li:before {
    content: "•";
    position: absolute;
    left: 0;
    color: rgba(255, 255, 255, 0.7);
}

/* Login section styling */
[data-testid="stSidebar"] .stButton > button {
background: linear-gradient(135deg, #1995AD, #085066) !important;
    color: white !important;
    font-weight: 600;
    border: none;
    border-radius: 20px;
    padding: 6px 20px;
    transition: all 0.3s;
    width: 100%;
    margin-top: 10px;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* Text input in sidebar */
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.4);
    color: white;
    border-radius: 5px;
}

[data-testid="stSidebar"] .stTextInput > label {
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Radio buttons in sidebar */
[data-testid="stSidebar"] .stRadio label {
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
}

[data-testid="stSidebar"] .stRadio {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
}

/* Remove info box styling, make it blend with the gradient */
[data-testid="stSidebar"] .stAlert {
    background-color: transparent !important;
    color: white !important;
    border-left: none !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] .stAlert p {
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
}
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 class="gradient-title">CORPORATE CLAUSE RISK ANALYZER</h1>
            <p class="subtitle">NLP-Powered Contract Analysis Tool</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    print("Authenticated?", st.session_state.get("authenticated"))


    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        login_or_signup()
        return
    
    # Initialize session state if necessary
    if "results" not in st.session_state:
        st.session_state.results = None
    
    if "show_chart" not in st.session_state:
        st.session_state.show_chart = False
    
    # File upload section
    with st.container():
        st.subheader("Upload Contract Document")
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Choose a PDF, DOCX or TXT file", type=["pdf", "docx", "txt"])
        with col2:
            st.markdown("<div style='height: 100px; display: flex; align-items: center;'>", unsafe_allow_html=True)
            button_placeholder = st.container()
            with button_placeholder:
                st.markdown(
                """
                <style>
                    .stButton > button {
                        width: 100% !important;
                    }
                </style>
                """,
                unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
         
    # Sample text input (alternative to file upload)
    use_sample = st.checkbox("Or use sample text input instead")
    sample_text = ""
    
    if use_sample:
        sample_text = st.text_area("Enter contract text here:", height=300)
    
    # Process button
    if uploaded_file is not None or (use_sample and sample_text):
        if st.button("Analyze Document"):
            with st.spinner("Analyzing document using NLP..."):
                try:
                    # Get text from either file or sample input
                    if not use_sample:
                        doc_text = extract_text_from_file(uploaded_file)
                        if not doc_text:
                            st.error("Could not extract text from the document. Please try another file.")
                            return
                    else:
                        doc_text = sample_text
                    
                    # Run analysis and store results in session state
                    st.session_state.results = analyze_document(doc_text)
                    
                    if "error" in st.session_state.results:
                        st.error(st.session_state.results["error"])
                        return
                    

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    # Clause Analysis (Filtered)
    if st.session_state.results:
        # Display results
        st.success("Analysis complete!")
        
        # Show risk summary in columns
       # Where you currently have the risk summary columns (col1, col2, col3, col4)
# Replace that section with this:

        st.markdown("<h3>Risk Analysis Summary</h3>", unsafe_allow_html=True)

# Create a nicer looking risk summary with cards
        st.markdown("""
        <div style="display:flex; justify-content:space-between; gap:15px; margin:10px 0 20px 0;">
            <div style="flex:1; background:linear-gradient(to bottom right, #fff, #f8f8f8); border-radius:10px; 
            box-shadow:0 4px 6px rgba(0,0,0,0.1); padding:15px; border-left:5px solid red;">
                <h3 style="margin:0; color:#1995AD; font-size:16px;">High Risk</h3>
                <p style="font-size:28px; font-weight:bold; margin:10px 0 0 0; color:red;">{}</p>
            </div>
            <div style="flex:1; background:linear-gradient(to bottom right, #fff, #f8f8f8); border-radius:10px; 
            box-shadow:0 4px 6px rgba(0,0,0,0.1); padding:15px; border-left:5px solid orange;">
                <h3 style="margin:0; color:#1995AD; font-size:16px;">Medium Risk</h3>
                <p style="font-size:28px; font-weight:bold; margin:10px 0 0 0; color:orange;">{}</p>
            </div>
            <div style="flex:1; background:linear-gradient(to bottom right, #fff, #f8f8f8); border-radius:10px; 
            box-shadow:0 4px 6px rgba(0,0,0,0.1); padding:15px; border-left:5px solid green;">
                <h3 style="margin:0; color:#1995AD; font-size:16px;">Low Risk</h3>
                <p style="font-size:28px; font-weight:bold; margin:10px 0 0 0; color:green;">{}</p>
            </div>
            <div style="flex:1; background:linear-gradient(to bottom right, #fff, #f8f8f8); border-radius:10px; 
            box-shadow:0 4px 6px rgba(0,0,0,0.1); padding:15px; border-left:5px solid #1995AD;">
                <h3 style="margin:0; color:#1995AD; font-size:16px;">Total Clauses</h3>
                <p style="font-size:28px; font-weight:bold; margin:10px 0 0 0; color:#1995AD;">{}</p>
            </div>
        </div>
        """.format(
            st.session_state.results["high_risk_count"],
            st.session_state.results["medium_risk_count"],
            st.session_state.results["low_risk_count"],
            st.session_state.results["total_clauses"]
        ), unsafe_allow_html=True)

        st.subheader("Risk Distribution")

        if st.button("View Risk Dashboard"):
            # Toggle the boolean flag in session state
            st.session_state.show_chart = not st.session_state.show_chart
        
        # Conditionally display the risk distribution chart based on the toggle
        if st.session_state.show_chart:
                
                risk_data = {
                "Risk Level": ["High", "Medium", "Low"],
                "Count": [st.session_state.results["high_risk_count"], 
                    st.session_state.results["medium_risk_count"], 
                    st.session_state.results["low_risk_count"]]
                    }
                risk_df = pd.DataFrame(risk_data)
                # Calculate the percentage for each risk level
                total_count = risk_df["Count"].sum()  # Get the total count of all risk levels
                risk_df["Percentage"] = (risk_df["Count"] / total_count) * 100  # Calculate the percentage

                # 📊 Interactive Bar Chart with Tooltips
                st.subheader("📊 Interactive Bar Chart with Tooltips")
                fig_bar = px.bar(
                    risk_df,
                    x="Risk Level",
                    y="Count",
                    color="Risk Level",
                    text="Count",
                    hover_data={"Risk Level": True, "Count": True, "Percentage": True},
                    color_discrete_map={"High": "#FF4B4B", "Medium": "#FFA500", "Low": "#2ECC71"},
                    title="Risk Level Distribution (Bar Chart)"
                )
                fig_bar.update_traces(
                    textposition='outside',
                    marker_line_color='black',
                    marker_line_width=1.5,
                    opacity=0.85
                )
                fig_bar.update_layout(
                    title_font=dict(size=20, color='#1995AD'),
                    font=dict(family="Segoe UI", size=14, color="#444"),
                    plot_bgcolor="#f9f9f9",
                    paper_bgcolor="#f1f9fb",
                    xaxis=dict(title=None),
                    yaxis=dict(title="Number of Clauses"),
                    margin=dict(l=30, r=30, t=60, b=40),
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # 🍩 Donut Chart of Risk Levels
                st.subheader("🍩 Donut Chart of Risk Levels")
                fig_donut = px.pie(
                    risk_df,
                    names="Risk Level",
                    values="Count",
                    color="Risk Level",
                    hole=0.5,
                    hover_data=["Percentage"],
                    color_discrete_map={"High": "#FF4B4B", "Medium": "#FFA500", "Low": "#2ECC71"},
                    title="Risk Level Proportions"
                )
                fig_donut.update_traces(textinfo="label+percent", pull=[0.05, 0.02, 0.02])
                fig_donut.update_layout(
                    title_font=dict(size=20, color='#1995AD'),
                    font=dict(family="Segoe UI", size=14),
                    margin=dict(l=20, r=20, t=60, b=40),
                    height=400
                )
                st.plotly_chart(fig_donut, use_container_width=True)

               

        # # Risk distribution chart
        # risk_data = {
        #     "Risk Level": ["High", "Medium", "Low"],
        #     "Count": [st.session_state.results["high_risk_count"], st.session_state.results["medium_risk_count"], st.session_state.results["low_risk_count"]]
        # }
        # risk_df = pd.DataFrame(risk_data)
        # st.subheader("Risk Distribution")
        # st.bar_chart(risk_df.set_index("Risk Level"))
        st.subheader("Clause Analysis")
        risk_filter = st.selectbox(
            "Filter by risk level:",
            ["All clauses", "High risk only", "Medium risk only", "Low risk only"]
        )
        
        # Apply filter
        filtered_clauses = st.session_state.results["clauses"]
        if risk_filter == "High risk only":
            filtered_clauses = [c for c in st.session_state.results["clauses"] if c["risk_level"] == "high"]
        elif risk_filter == "Medium risk only":
            filtered_clauses = [c for c in st.session_state.results["clauses"] if c["risk_level"] == "medium"]
        elif risk_filter == "Low risk only":
            filtered_clauses = [c for c in st.session_state.results["clauses"] if c["risk_level"] == "low"]
        
        # Display clauses
        for i, clause in enumerate(filtered_clauses):
            clause_type_formatted = clause["type"].replace("_", " ").title()
            risk_colors = {"high": "red", "medium": "orange", "low": "green"}
            risk_level = clause["risk_level"]
            risk_color = risk_colors[risk_level]

            with st.container():
                st.markdown(f"""
                    <div class='risk-box'>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: bold; color: #1995AD; font-size: 16px;">{clause_type_formatted}</span>
                            <span class="risk-indicator {risk_color}">{risk_level.upper()} RISK</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
                with st.expander("View Details"):
                    st.markdown("<div class='expander-content'>", unsafe_allow_html=True)
                    st.markdown(f"**Summary:** {clause['summary']}")
                    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                    st.markdown("**Full Text:**")
                    st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{clause['text']}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
# Add Export Results functionality
        import json
        import io

        # Export Results section
        if st.session_state.results:
            st.subheader("Export Analysis Results")

            # Convert results to JSON string
            json_data = json.dumps(st.session_state.results, indent=2)
            st.download_button(
                label="📥 Download Full Analysis (JSON)",
                data=json_data,
                file_name="clause_analysis.json",
                mime="application/json"
            )

            # Convert clause data to CSV using pandas
            clauses_df = pd.DataFrame(st.session_state.results["clauses"])
            csv_buffer = io.StringIO()
            clauses_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Clause Table (CSV)",
                data=csv_buffer.getvalue(),
                file_name="clause_analysis.csv",
                mime="text/csv"
            )
    # Sidebar with information about the tool
    with st.sidebar:
        if st.session_state.get("authenticated"):
            st.markdown(f"**Logged in as:** {st.session_state['username']}")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.rerun()
        st.title("📜 About")
        st.markdown(
        """
        This tool uses Natural Language Processing (NLP) to analyze corporate contracts 
        and identify potential risks. It can:
        
        • Identify different types of clauses 
        • Assess risk levels based on legal language 
        • Summarize clause content 
        • Highlight areas that may need legal review 
        
        Powered by advanced NLP models including BERT and Transformer-based summarization.
        """,
                unsafe_allow_html=True
        )
        
        st.subheader("Clause Types Detected")
        st.markdown(
        """
        - 🛡️ Indemnification
        - ⚖️ Limitation of Liability
        - 🚫 Termination
        - 🔒 Confidentiality
        - 💳 Payment Terms
        - ⚙️ Warranty
        - 💡 Intellectual Property
        - ⚖️ Governing Law
        - 🌩️ Force Majeure
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()