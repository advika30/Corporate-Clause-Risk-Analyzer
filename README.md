# Corporate-Clause-Risk-Analyzer

Corporate Clause Risk Analyzer 📑
A Streamlit-based Natural Language Processing (NLP) application to analyze corporate contracts, classify clauses, and assess legal risk levels automatically.
Powered by BERT embeddings, Transformer-based summarization, and customizable legal clause detection.

Features
📄 Upload Contracts: Supports PDF, DOCX, and TXT formats.

🔍 Clause Detection: Splits contracts into individual clauses intelligently.

🏷️ Clause Classification: Identifies clause types like Indemnification, Termination, Payment Terms, etc.

⚡ Risk Analysis: Rates clauses into High, Medium, or Low risk based on NLP-driven pattern detection.

✨ Automatic Summarization: Summarizes long legal clauses using Transformer models.

📊 Visual Dashboard: Displays risk distribution and detailed clause reports in an intuitive UI.

Tech Stack 🛠️
Frontend: Streamlit

NLP Models:

BERT (Sentence Embeddings)

Facebook BART Large CNN (Summarization)

spaCy (Named Entity Recognition)

Machine Learning: Random Forest, TF-IDF

Other Libraries: Scikit-learn, Pandas, Transformers, Torch, NLTK, pdfminer.six, docx2txt

Setup Instructions 🚀
Clone this repository:

bash
Copy
Edit
git clone https://github.com/your-username/corporate-clause-risk-analyzer.git
cd corporate-clause-risk-analyzer
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Download necessary spaCy model:

bash
Copy
Edit
python -m spacy download en_core_web_sm
Run the app:

bash
Copy
Edit
streamlit run bert2.py
Demo Screenshots 📸
(Add your screenshots here after hosting locally — e.g., Upload Section, Risk Summary, Clause Details)

Future Improvements 🔥
Add contract comparison feature (highlight differences clause-by-clause).

Provide automated rewriting suggestions for high-risk clauses.

Enable multilingual contract support (Spanish, German, etc).

Deploy SaaS version with user authentication.

License 📄
This project is open-sourced under the MIT License. Feel free to use and adapt!
