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
        body {
            background-color: #F1F1F2;
            color: #1995AD;
        }
        .stApp {
             background: linear-gradient(to bottom right, #f1f9fb, #e4f2f6);
        }
        [data-testid="stSidebar"] {
            background-color:#ace2ef !important;
            color: #1995AD !important;
            border-left: 4px solid #1995AD;
        }
        

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

# Streamlit UI
def main():
    st.title("CORPORATE CLAUSE RISK ANALYZER")
    st.markdown("### NLP-Powered Contract Analysis Tool")
    
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
                    show_analysis_progress()
                    # Run analysis
                    results = analyze_document(doc_text)
                    
                    if "error" in results:
                        st.error(results["error"])
                        return
                    
                    # Display results
                    st.success("Analysis complete!")
                    
                    # Show risk summary in columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                      st.markdown(f"""
                         <div style="border-radius:10px; padding:10px; text-align:center; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
                          <p style="margin:0; color:#1995AD;">High Risk</p>
                          <p style="font-size:24px; font-weight:bold; margin:0; color:red;">{results["high_risk_count"]}</p>
                         </div>
                      """, unsafe_allow_html=True)
                    with col2:
                       st.markdown(f"""
                          <div style="border-radius:10px; padding:10px; text-align:center; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
                            <p style="margin:0; color:#1995AD;">Medium Risk</p>
                            <p style="font-size:24px; font-weight:bold; margin:0; color:orange;">{results["medium_risk_count"]}</p>
                          </div>
                       """, unsafe_allow_html=True)
                    with col3:
                       st.markdown(f"""
                           <div style="border-radius:10px; padding:10px; text-align:center; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
                               <p style="margin:0; color:#1995AD;">Low Risk</p>
                               <p style="font-size:24px; font-weight:bold; margin:0; color:green;">{results["low_risk_count"]}</p>
                           </div>
                       """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                            <div style="border-radius:10px; padding:10px; text-align:center; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
                              <p style="margin:0; color:#1995AD;">Total Clauses</p>
                              <p style="font-size:24px; font-weight:bold; margin:0; color:#1995AD;">{results["total_clauses"]}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk distribution chart
                    risk_data = {
                        "Risk Level": ["High", "Medium", "Low"],
                        "Count": [results["high_risk_count"], results["medium_risk_count"], results["low_risk_count"]]
                    }
                    risk_df = pd.DataFrame(risk_data)
                    st.subheader("Risk Distribution")
                    st.bar_chart(risk_df.set_index("Risk Level"))
                    
                    # Filter controls
                    st.subheader("Clause Analysis")
                    risk_filter = st.selectbox(
                        "Filter by risk level:",
                        ["All clauses", "High risk only", "Medium risk only", "Low risk only"]
                    )
                    
                    # Apply filter
                    filtered_clauses = results["clauses"]
                    if risk_filter == "High risk only":
                        filtered_clauses = [c for c in results["clauses"] if c["risk_level"] == "high"]
                    elif risk_filter == "Medium risk only":
                        filtered_clauses = [c for c in results["clauses"] if c["risk_level"] == "medium"]
                    elif risk_filter == "Low risk only":
                        filtered_clauses = [c for c in results["clauses"] if c["risk_level"] == "low"]
                    # Display clauses
                    for i, clause in enumerate(filtered_clauses):
                    # Format the clause type for display
                      clause_type_formatted = clause["type"].replace("_", " ").title()
    
                      # Define risk level colors
                      risk_colors = {
                           "high": "red",
                           "medium": "orange", 
                           "low": "green"
                             }
    
                      risk_level = clause["risk_level"]
                      risk_color = risk_colors[risk_level]
    
                      # Create expandable card for each clause with colored risk level
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
                
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    # Add information about the tool
    with st.sidebar:
        st.title("About")
        st.info(
            """
            This tool uses Natural Language Processing (NLP) to analyze corporate contracts 
            and identify potential risks. It can:
            
            - Identify different types of clauses
            - Assess risk levels based on legal language
            - Summarize clause content
            - Highlight areas that may need legal review
            
            Powered by advanced NLP models including BERT and Transformer-based summarization.
            """
        )
        
        st.subheader("Clause Types Detected")
        st.markdown(
            """
            - Indemnification
            - Limitation of Liability
            - Termination
            - Confidentiality
            - Payment Terms
            - Warranty
            - Intellectual Property
            - Governing Law
            - Force Majeure
            """
        )

if __name__ == "__main__":
    main()