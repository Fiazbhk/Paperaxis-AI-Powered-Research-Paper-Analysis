# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from flask import Flask, render_template, request, jsonify
#
# from src.load_and_extract_text import extract_text_from_pdf, extract_pdf_sections
# from src.detect_and_split_sections import refine_sections, split_sections_with_content
# from src.get_summary import generate_detailed_summary
# from src.create_vector_db import create_vector_db
# from src.RAG_retrival_chain import get_qa_chain
#
# from dotenv import load_dotenv
# import os, json
#
# load_dotenv()
#
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
#
# # Ensure upload folder exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#
# groq_api_key = os.getenv("GROQ_API_KEY")
# llm_model = os.getenv("LLM_MODEL")
# embedding_model = os.getenv("EMBEDDING_MODEL")
#
# # Global variable
# full_text = ''
# Research_paper_topics = None
# vector_db = None
#
# llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
#
# # Initialize embeddings using the Hugging Face model
# embedder = HuggingFaceEmbeddings(model_name=embedding_model)
#
#
# # print(llm.invoke("why diwali celebrate ?").content)
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     global full_text
#     global Research_paper_topics
#
#     file = request.files.get('file')
#
#     if not file:
#         return jsonify({"error": "No file uploaded"}), 400
#
#     filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     # print(filename)
#     file.save(filename)
#
#     # Get all topics name from research paper
#     extracted_text = extract_text_from_pdf(filename)
#     full_text = extracted_text
#     extracted_sections = extract_pdf_sections(full_text=extracted_text)
#     # print(extracted_sections)
#     refined_sections = refine_sections(extracted_sections, llm)
#     # print(refined_sections)
#     section_with_content = split_sections_with_content(extracted_text, refined_sections)
#
#     Research_paper_topics = section_with_content
#
#     return jsonify({"topics": list(Research_paper_topics.keys())})
#
#
# @app.route('/summary', methods=['POST'])
# def get_summary():
#     global Research_paper_topics
#
#     topic = request.json.get('topic')
#     # print(topic)
#
#     topic_content = Research_paper_topics.get(topic, "No summary available.")
#
#     summary = generate_detailed_summary(topic_content, llm)
#
#     return jsonify({"summary": summary})
#
#
# @app.route('/chat', methods=['POST'])
# def chat():
#     global full_text
#     global vector_db
#
#     user_message = request.json.get('message')
#     print(user_message)
#
#     if not vector_db:
#         vectordb = create_vector_db(text=full_text, embedder=embedder)
#         vector_db = vectordb
#
#     chain = get_qa_chain(vectordb=vector_db, llm=llm)
#
#     ai_response = chain.invoke(user_message)['result']
#     print(ai_response)
#
#     return jsonify({"response": ai_response})
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
#
#     # extracted_text = extract_text_from_pdf("paper.pdf")
#     # print(extracted_text)
#     # extracted_sections = extract_pdf_sections(full_text = extracted_text)
#     # # with open("extracted_sections.json", "w") as f:
#     # #     json.dump(extracted_sections, f, indent=4)
#
#     # refined_sections = refine_sections(extracted_sections, llm)
#     # # with open("refined_sections.json", "w") as f:
#     # #     json.dump(refined_sections, f, indent=4)
#
#     # section_with_content = split_sections_with_content(extracted_text, refined_sections)
#     # with open("section_with_content.json", "w") as f:
#     #     json.dump(section_with_content, f, indent=4)

import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.load_and_extract_text import extract_text_from_pdf, extract_pdf_sections
from src.detect_and_split_sections import refine_sections, split_sections_with_content
from src.get_summary import generate_detailed_summary
from src.create_vector_db import create_vector_db
from src.RAG_retrival_chain import get_qa_chain

from dotenv import load_dotenv
from streamlit_option_menu import option_menu

# -----------------------------
# ENV + MODELS
# -----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm_model = os.getenv("LLM_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=llm_model
)

embedder = HuggingFaceEmbeddings(
    model_name=embedding_model
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# SESSION STATE
# -----------------------------
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

if "sections" not in st.session_state:
    st.session_state.sections = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# -----------------------------
# 1. TITLE
# -----------------------------
st.set_page_config(layout="centered")

st.sidebar.title("❉ Paperaxis")

st.sidebar.markdown(
    """
**AI-powered Research Paper Analysis**

This project leverages AI and NLP to analyze research papers, generate summaries,
extract keywords, and visualize trends.

**Tech Stack:** Python, AI / ML, NLP, PDF Parsing, Streamlit
"""
)

st.sidebar.markdown("### Key Features")

st.sidebar.markdown(
    """
- Upload and analyze research PDFs
- Extract sections automatically from paper
- Generate summaries per selected section
- Chat with paper using AI
- Restart chat to clear history
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
[GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile) | [LeetCode](https://leetcode.com/your-profile)
"""
)

st.title("❉ Paperaxis")

# -----------------------------
# Horizontal Menu (Tabs)
# -----------------------------
selected_tab = option_menu(
    menu_title=None,
    options=["Get Summary", "Chat with Paper"],
    icons=["file-text", "chat"],
    orientation="horizontal",
    default_index=0
)

# -----------------------------
# GET SUMMARY TAB
# -----------------------------
if selected_tab == "Get Summary":
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])

    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        extracted_text = extract_text_from_pdf(file_path)
        st.session_state.full_text = extracted_text

        extracted_sections = extract_pdf_sections(full_text=extracted_text)
        refined_sections = refine_sections(extracted_sections, llm)
        section_with_content = split_sections_with_content(
            extracted_text,
            refined_sections
        )

        st.session_state.sections = section_with_content
        st.session_state.vector_db = None

        st.success("PDF processed successfully")

    if st.session_state.sections:
        st.subheader("Section Summary")

        section_names = list(st.session_state.sections.keys())
        selected_section = st.selectbox(
            "Select Section",
            section_names,
            key="selected_section"
        )

        if st.button("Generate Summary"):
            content = st.session_state.sections[st.session_state.selected_section]
            summary = generate_detailed_summary(content, llm)
            st.write(summary)

# -----------------------------
# CHAT TAB (Enhanced Chat UI)
# -----------------------------
if selected_tab == "Chat with Paper":
    st.subheader("Chat with Research Paper")

    if not st.session_state.full_text:
        st.warning("Upload a paper first in the Get Summary tab.")
    else:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat input
        user_message = st.chat_input("Ask a question...")

        # Display previous messages
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_message:
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_message)

            # Generate response using your RAG chain
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if not st.session_state.vector_db:
                        st.session_state.vector_db = create_vector_db(
                            text=st.session_state.full_text,
                            embedder=embedder
                        )

                    chain = get_qa_chain(
                        vectordb=st.session_state.vector_db,
                        llm=llm
                    )

                    response = chain.invoke(user_message)["result"]
                    st.markdown(response)

            # Append messages to session state
            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Restart button to clear chat history
        def clear_chat():
            st.session_state.messages = []

        st.button("Restart Chat", on_click=clear_chat)


# import streamlit as st
# import os
#
# from streamlit_option_menu import option_menu
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
#
# from src.load_and_extract_text import extract_text_from_pdf, extract_pdf_sections
# from src.detect_and_split_sections import refine_sections, split_sections_with_content
# from src.get_summary import generate_detailed_summary
# from src.create_vector_db import create_vector_db
# from src.RAG_retrival_chain import get_qa_chain
#
# from dotenv import load_dotenv
#
# # --------------------------------------------------
# # PAGE CONFIG
# # --------------------------------------------------
# st.set_page_config(
#     page_title="Paperaxis",
#     layout="wide"
# )
#
# # --------------------------------------------------
# # SIDEBAR
# # --------------------------------------------------
# st.sidebar.title("Paperaxis")
#
# st.sidebar.markdown(
#     """
# AI-powered Research Paper Analysis
#
# This project leverages AI and NLP to analyze research papers, generate summaries,
# extract keywords, and visualize trends.
#
# **Tech Stack:** Python, AI / ML, NLP, PDF Parsing, Streamlit
# """
# )
#
# st.sidebar.markdown(
#     """
# **Key Features**
# - PDF Analysis
# - Summary Generation
# - Topic Modeling
# - Citation Extraction
# """
# )
#
# st.sidebar.markdown("---")
# st.sidebar.markdown(
#     """
# GitHub | LinkedIn | LeetCode
# """
# )
#
# # --------------------------------------------------
# # ENV + MODELS
# # --------------------------------------------------
# load_dotenv()
#
# llm = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model_name=os.getenv("LLM_MODEL")
# )
#
# embedder = HuggingFaceEmbeddings(
#     model_name=os.getenv("EMBEDDING_MODEL")
# )
#
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
#
# # --------------------------------------------------
# # SESSION STATE
# # --------------------------------------------------
# if "full_text" not in st.session_state:
#     st.session_state.full_text = None
#
# if "sections" not in st.session_state:
#     st.session_state.sections = None
#
# if "vector_db" not in st.session_state:
#     st.session_state.vector_db = None
#
# # --------------------------------------------------
# # MAIN TITLE
# # --------------------------------------------------
# st.title("Research Agent")
#
# # --------------------------------------------------
# # HORIZONTAL MENU (REPLACES TABS)
# # --------------------------------------------------
# selected = option_menu(
#     menu_title=None,
#     options=["Get Summary", "Chat with Paper"],
#     icons=["file-text", "chat"],
#     orientation="horizontal",
#     default_index=0
# )
#
# # ==================================================
# # GET SUMMARY PAGE
# # ==================================================
# if selected == "Get Summary":
#     st.subheader("Upload Research Paper")
#
#     uploaded_file = st.file_uploader(
#         "Upload PDF",
#         type=["pdf"]
#     )
#
#     if uploaded_file:
#         file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
#
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#
#         extracted_text = extract_text_from_pdf(file_path)
#         st.session_state.full_text = extracted_text
#
#         extracted_sections = extract_pdf_sections(
#             full_text=extracted_text
#         )
#
#         refined_sections = refine_sections(
#             extracted_sections,
#             llm
#         )
#
#         st.session_state.sections = split_sections_with_content(
#             extracted_text,
#             refined_sections
#         )
#
#         st.session_state.vector_db = None
#
#         st.success("PDF processed successfully")
#
#     if st.session_state.sections:
#         section_names = list(st.session_state.sections.keys())
#
#         selected_section = st.selectbox(
#             "Select Section",
#             section_names
#         )
#
#         if st.button("Generate Summary"):
#             summary = generate_detailed_summary(
#                 st.session_state.sections[selected_section],
#                 llm
#             )
#             st.write(summary)
#
# # ==================================================
# # CHAT PAGE
# # ==================================================
# if selected == "Chat with Paper":
#     st.subheader("Chat with Research Paper")
#
#     if not st.session_state.full_text:
#         st.warning("Upload a paper first from Get Summary.")
#     else:
#         query = st.text_input("Ask a question")
#
#         if query:
#             if not st.session_state.vector_db:
#                 st.session_state.vector_db = create_vector_db(
#                     text=st.session_state.full_text,
#                     embedder=embedder
#                 )
#
#             chain = get_qa_chain(
#                 vectordb=st.session_state.vector_db,
#                 llm=llm
#             )
#
#             response = chain.invoke(query)["result"]
#             st.write(response)

