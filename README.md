# Paperaxis — AI-Powered Research Paper Analysis

Paperaxis is a Final Year Project focused on automated understanding of research papers using Artificial Intelligence and Natural Language Processing. The system allows users to upload a research paper in PDF format, generate section-wise summaries, and interactively query the paper using an AI-driven chat interface.

The project targets students and researchers who need faster comprehension of technical literature.

---

## Problem Statement

Research papers are lengthy, complex, and time-intensive to analyze manually. Extracting summaries, understanding section intent, and locating relevant information requires significant effort. This project addresses the problem by automating paper analysis and enabling natural-language interaction with the document.

---

## Objectives

- Automate PDF text extraction  
- Detect and segment paper sections  
- Generate section-wise summaries  
- Enable document-aware question answering  
- Provide an interactive web interface  

---

## Key Features

- Research paper PDF upload  
- Automatic section identification  
- AI-generated section summaries  
- Chat with paper content  
- RAG-based semantic retrieval  
- Session-persistent interactions  

---

## System Overview

1. **PDF Ingestion**  
   Extracts raw text from uploaded research papers.

2. **Section Detection**  
   Identifies logical sections such as Abstract, Introduction, Methodology, and Conclusion.

3. **Summarization Engine**  
   Uses a Large Language Model to generate detailed summaries for selected sections.

4. **RAG-Based Chat Module**  
   - Splits document text into chunks  
   - Converts chunks into embeddings  
   - Stores embeddings in FAISS  
   - Retrieves relevant context for answering user queries  

5. **User Interface**  
   Implemented using Streamlit with menu-based navigation.

---

## Technology Stack

- **Language:** Python  
- **Frontend:** Streamlit  
- **LLM Framework:** LangChain  
- **LLM Provider:** Groq  
- **Embeddings:** HuggingFace Sentence Transformers  
- **Vector Store:** FAISS  
- **PDF Processing:** PyPDF2  

---

## Project Structure

<img width="232" height="244" alt="image" src="https://github.com/user-attachments/assets/49c12d82-752b-4fe1-8505-92849b77c4c6" />


---

## Installation

### Clone Repository
```bash
git clone https://github.com/<your-username>/paperaxis-ai-powered-research-paper-analysis.git
cd paperaxis-ai-powered-research-paper-analysis
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a .env file locally:
```bash
GROQ_API_KEY=your_api_key
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```
### Run Application
```bash
streamlit run app.py
```

## Deployment

The application is deployed using **Streamlit Cloud**.  
Sensitive information such as API keys is not stored in the repository and is managed securely using **Streamlit Secrets**.

To deploy:
1. Push the project to a GitHub repository.
2. Connect the repository to Streamlit Cloud.
3. Set required environment variables in **App Settings → Secrets**.
4. Select `app.py` as the entry point.
5. Deploy the application.

---

## Limitations

- Performance depends on the quality of the uploaded PDF  
- Scanned or image-based PDFs may yield poor results  
- Large documents increase processing and response time  
- AI-generated summaries may contain inaccuracies  

---

## Future Enhancements

- Multi-paper comparison and analysis  
- Reference and citation extraction  
- Export summaries and chat history  
- Support for DOCX and LaTeX files  
- Improved section detection accuracy  

---

## Academic Context

This project is developed as a **Final Year Project (FYP)** in partial fulfillment of the requirements for a Bachelor’s degree in Computer Science / Information Technology.

---

## Author

**Muhammad Fiaz**  
Final Year Undergraduate  
Department of Computer Science & Information Technology 
University of Sargodha

---

## License

Apache License 2.0

